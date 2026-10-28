"""Tissue patch datasets with fixed ontology and disjoint section partitions."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import IMAGE_MEAN, IMAGE_STD, _coordinates, unique_file
from src.contracts import LABEL2ID, assert_disjoint_sections


def get_transforms(mode='train'):
    operations = []
    if mode == 'train':
        operations = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    return transforms.Compose(operations + [transforms.ToTensor(),
                              transforms.Normalize(IMAGE_MEAN.flatten().tolist(),
                                                   IMAGE_STD.flatten().tolist())])


class SpotFormerDataset(Dataset):
    def __init__(self, data_root, samples, patch_size=224, max_spots_per_sample=None,
                 mode='train', auto_patch_size=False, skip_undetermined=True,
                 undersample_majority=False, undersample_ratio=0.5, val_spots_file=None,
                 label_to_idx=None, records=None, transform=None):
        if patch_size != 224 or auto_patch_size:
            raise ValueError('The manuscript extracts fixed 224x224-pixel patches.')
        if val_spots_file:
            raise ValueError('Spot-level validation splits are prohibited; use separate sections.')
        if max_spots_per_sample is not None:
            raise ValueError('All section spots are retained; chunk the forward pass instead of subsampling.')
        if undersample_majority:
            raise ValueError('The section protocol retains all annotated spots without undersampling.')
        self.data_root, self.samples, self.mode = Path(data_root), list(samples), mode
        self.patch_size = patch_size
        self.label_to_idx = dict(label_to_idx if label_to_idx is not None else LABEL2ID)
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        self.transform = transform or get_transforms(mode)
        self.sample_data = []
        candidates = [self.data_root, self.data_root / 'her2st' / 'data', self.data_root / 'data']
        base = next((p for p in candidates if (p / 'ST-imgs').exists()), self.data_root)
        for section in self.samples:
            if records is None:
                image_path = unique_file(
                    [p for p in (base / 'ST-imgs' / section[0] / section).glob('*')
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')],
                    f'image for {section}')
                annotation_path = base / 'ST-pat' / 'lbl' / f'{section}_labeled_coordinates.tsv'
                position_path = base / 'ST-spotfiles' / f'{section}_selection.tsv'
            else:
                record = records[section]
                image_path, position_path = Path(record['image']), Path(record['positions'])
                annotation_path = Path(record['labels']) if record.get('labels') else None
            if position_path.exists():
                spots = _coordinates(pd.read_csv(position_path, sep='\t'))
            elif annotation_path is not None and annotation_path.exists():
                spots = _coordinates(pd.read_csv(annotation_path, sep='\t'))
            else:
                raise FileNotFoundError(f'No coordinates for {section}.')
            spots = spots.copy()
            spots['label'] = 'undetermined'
            if annotation_path is not None and annotation_path.exists():
                labels = pd.read_csv(annotation_path, sep='\t')
                labels.index = [f'{x}x{y}' for x, y in
                                labels[['x', 'y']].round().astype(int).to_numpy()]
                if labels.index.duplicated().any():
                    raise ValueError(f'Duplicate annotations in {section}.')
                spots['label'] = labels['label'].reindex(spots.index).fillna('undetermined')
            spots['label'] = spots['label'].astype(str).str.strip().str.lower()
            spots = spots.loc[np.isfinite(spots[['pixel_x', 'pixel_y']]).all(axis=1)]
            # Supervised classification ignores unknown labels; inference keeps every spot.
            if skip_undetermined and mode != 'inference':
                spots = spots.loc[spots['label'].isin(self.label_to_idx)]
            if not len(spots):
                raise ValueError(f'No usable spots in section {section}.')
            self.sample_data.append(dict(sample_id=section, image_path=str(image_path),
                                         spots_data=spots))

    def __len__(self):
        return len(self.sample_data)

    def __getitem__(self, index):
        item = self.sample_data[index]
        spots = item['spots_data']
        patches = []
        with Image.open(item['image_path']) as source:
            image = source.convert('RGB')
            for x, y in spots[['pixel_x', 'pixel_y']].to_numpy():
                x, y = int(round(x)), int(round(y))
                patch = image.crop((x - 112, y - 112, x + 112, y + 112))
                patches.append(self.transform(patch))
        return dict(patches=torch.stack(patches),
                    labels=torch.tensor([self.label_to_idx.get(label, -1)
                                         for label in spots['label']], dtype=torch.long),
                    positions=torch.from_numpy(spots[['pixel_x', 'pixel_y']].to_numpy(np.float32)),
                    spatial_coords=torch.from_numpy(spots[['x', 'y']].to_numpy(np.float32)),
                    spot_ids=spots.index.tolist(), mask=torch.ones(len(spots), dtype=torch.bool),
                    sample_id=item['sample_id'], num_spots=len(spots))

    @staticmethod
    def collate_fn(batch):
        if not batch:
            raise ValueError('Empty section batch.')
        # One complete section is the iteration unit; no cross-section padding/subsampling.
        if len(batch) != 1:
            raise ValueError('The manuscript uses batch_size=1 tissue section.')
        row = batch[0]
        return dict(patches=row['patches'][None], labels=row['labels'][None],
                    positions=row['positions'][None], spatial_coords=row['spatial_coords'][None],
                    mask=row['mask'][None], sample_ids=[row['sample_id']],
                    spot_ids=[row['spot_ids']], num_spots=torch.tensor([row['num_spots']]))


def create_dataloaders(data_root, train_samples, val_samples=None, test_samples=None,
                       batch_size=1, max_spots_per_sample=None, num_workers=0, patch_size=224,
                       auto_patch_size=False, skip_undetermined=True, validation_split=0.1,
                       undersample_majority=False, undersample_ratio=0.5, seed=42,
                       label_to_idx=None, records=None):
    train_samples, test_samples = list(train_samples), list(test_samples or [])
    if val_samples is None:
        shuffled = train_samples[:]
        random.Random(seed).shuffle(shuffled)
        count = max(1, round(len(shuffled) * (validation_split or 0.1)))
        val_samples, train_samples = shuffled[:count], shuffled[count:]
    val_samples = list(val_samples)
    assert_disjoint_sections(train_samples, val_samples, test_samples)
    if not train_samples or not val_samples:
        raise ValueError('Classifier training and validation require separate nonempty sections.')
    if batch_size != 1:
        raise ValueError('Use one section per iteration and <=128 spot chunks.')
    ontology = dict(label_to_idx if label_to_idx is not None else LABEL2ID)
    loaders = {}
    for key, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        if not samples:
            continue
        dataset = SpotFormerDataset(data_root, samples, patch_size, max_spots_per_sample,
                                    mode='train' if key == 'train' else 'test',
                                    auto_patch_size=auto_patch_size,
                                    skip_undetermined=skip_undetermined,
                                    undersample_majority=undersample_majority,
                                    undersample_ratio=undersample_ratio, label_to_idx=ontology,
                                    records=records)
        loaders[key] = DataLoader(dataset, batch_size=1, shuffle=key == 'train',
                                  num_workers=num_workers, collate_fn=SpotFormerDataset.collate_fn)
    return loaders, ontology, {i: label for label, i in ontology.items()}

