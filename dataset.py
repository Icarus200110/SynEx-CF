"""Section-level datasets with training-only gene selection and full references."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.contracts import LABEL2ID, split_sections

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def unique_file(paths, description):
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f'Expected one {description}; found {len(paths)}: {paths}')
    return paths[0]


def discover_sections(data_root='data', dataset='her2st', manifest=None):
    """An optional JSON manifest maps section IDs to counts/positions/image/labels paths."""
    root = Path(data_root).resolve()
    if manifest:
        manifest_path = Path(manifest).resolve()
        records = json.loads(manifest_path.read_text(encoding='utf-8'))
        records = records.get('sections', records)
        if isinstance(records, list):
            records = {r['section']: r for r in records}
        return {name: {key: str((manifest_path.parent / value).resolve())
                       for key, value in row.items()
                       if key in ('counts', 'positions', 'image', 'labels')}
                for name, row in records.items()}
    records = {}
    if dataset.lower() == 'her2st':
        candidates = [root, root / 'her2st' / 'data', root / 'data']
        base = next((p for p in candidates if (p / 'ST-cnts').is_dir()), None)
        if base is None:
            raise FileNotFoundError(f'Cannot find HER2ST ST-cnts under {root}')
        for count_file in sorted((base / 'ST-cnts').glob('*.tsv')):
            section = count_file.stem
            image = unique_file([p for p in (base / 'ST-imgs' / section[0] / section).glob('*')
                                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')],
                                f'image for {section}')
            record = dict(counts=str(count_file),
                          positions=str(base / 'ST-spotfiles' / f'{section}_selection.tsv'),
                          image=str(image))
            labels = base / 'ST-pat' / 'lbl' / f'{section}_labeled_coordinates.tsv'
            if labels.exists():
                record['labels'] = str(labels)
            records[section] = record
    elif dataset.lower() == 'cscc':
        base = root if list(root.glob('*_stdata.tsv')) else root / 'GSE144240_RAW'
        # Discover all available patients/replicates rather than restricting to four patients.
        import re
        for count_file in sorted(base.glob('*_stdata.tsv')):
            match = re.search(r'(P\d+_ST_rep\d+)_stdata$', count_file.stem)
            if not match:
                raise ValueError(f'Unrecognized cSCC section identifier: {count_file.name}')
            section = match.group(1)
            record = dict(counts=str(count_file),
                          positions=str(unique_file(base.glob(f'*spot*{section}.tsv'),
                                                    f'coordinates for {section}')),
                          image=str(unique_file(base.glob(f'*{section}.jpg'),
                                                f'image for {section}')))
            labels = base / f'{section}_labeled_coordinates.tsv'
            if labels.exists():
                record['labels'] = str(labels)
            records[section] = record
    else:
        raise ValueError('Use her2st/cscc or provide an explicit JSON section manifest.')
    if not records:
        raise ValueError('No sections found.')
    return records


def _coordinates(frame):
    required = {'x', 'y', 'pixel_x', 'pixel_y'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing coordinate columns: {sorted(required - set(frame.columns))}')
    frame = frame.copy()
    frame[['x', 'y']] = frame[['x', 'y']].round().astype(int)
    frame.index = [f'{x}x{y}' for x, y in frame[['x', 'y']].to_numpy()]
    if frame.index.duplicated().any():
        raise ValueError('Duplicate physical spot coordinates.')
    return frame


def load_section(record):
    counts = pd.read_csv(record['counts'], sep='\t', index_col=0)
    counts.columns = counts.columns.astype(str)
    counts.index = counts.index.astype(str)
    if counts.columns.duplicated().any() or counts.index.duplicated().any():
        raise ValueError(f'Duplicate genes or spots in {record["counts"]}')
    counts = counts.apply(pd.to_numeric, errors='raise')
    values = counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expression inputs must be finite nonnegative raw counts.')
    positions = _coordinates(pd.read_csv(record['positions'], sep='\t'))
    common = counts.index.intersection(positions.index, sort=False)
    counts, positions = counts.loc[common], positions.loc[common]
    # Remove empty spots and invalid image coordinates before computing libraries.
    keep = ((counts.sum(axis=1) > 0)
            & np.isfinite(positions[['pixel_x', 'pixel_y']]).all(axis=1))
    counts, positions = counts.loc[keep], positions.loc[keep]
    if len(counts) < 6:
        raise ValueError('At least six nonempty spots are required per tissue section.')
    # Normalize the complete measured library before selecting any target genes.
    expression = np.log2(1 + counts.div(counts.sum(axis=1), axis=0) * 1e6).astype(np.float32)
    labels = pd.Series('undetermined', index=counts.index, dtype=object)
    if record.get('labels'):
        annotated = pd.read_csv(record['labels'], sep='\t')
        if not {'x', 'y', 'label'}.issubset(annotated.columns):
            raise ValueError('Annotation files must contain x, y and label.')
        annotated.index = [f'{x}x{y}' for x, y in
                           annotated[['x', 'y']].round().astype(int).to_numpy()]
        if annotated.index.duplicated().any():
            raise ValueError('Duplicate annotation coordinates.')
        labels = annotated['label'].reindex(counts.index).fillna('undetermined')
    return dict(expression=expression, positions=positions, labels=labels,
                image=record['image'])


def extract_patches(image_path, pixel_coordinates):
    """PIL crop pads image boundaries and always preserves 224x224 pixels."""
    with Image.open(image_path) as source:
        image = source.convert('RGB')
        for x, y in pixel_coordinates:
            x, y = int(round(x)), int(round(y))
            patch = np.array(image.crop((x - 112, y - 112, x + 112, y + 112)), copy=True)
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float().div_(255)
            yield (tensor - IMAGE_MEAN) / IMAGE_STD


class SpatialExpressionDataset(Dataset):
    def __init__(self, names, records, core_genes, full_gene_order, label_to_idx=None,
                 prepared=None):
        self.names = list(names)
        self.gene_set = list(core_genes)
        self.full_gene_order = list(full_gene_order)
        self.label_to_idx = dict(label_to_idx if label_to_idx is not None else LABEL2ID)
        self.sections = {name: prepared[name] if prepared is not None and name in prepared
                         else load_section(records[name]) for name in self.names}
        for name, section in self.sections.items():
            missing = set(self.full_gene_order) - set(section['expression'].columns)
            if missing:
                raise ValueError(f'{name} lacks {len(missing)} training-reference genes; '
                                 'supply harmonized expression matrices, do not fit on held-out data.')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        section = self.sections[name]
        positions = section['positions']
        expression = section['expression']
        patches = torch.stack(list(extract_patches(section['image'],
                                                   positions[['pixel_x', 'pixel_y']].to_numpy())))
        return dict(section=name, spot_ids=expression.index.tolist(), patches=patches,
                    positions=torch.from_numpy(positions[['x', 'y']].to_numpy(dtype=np.int64)),
                    centers=torch.from_numpy(positions[['pixel_x', 'pixel_y']].to_numpy(np.float32)),
                    expression=torch.from_numpy(expression[self.gene_set].to_numpy(copy=True)),
                    full_expression=torch.from_numpy(
                        expression[self.full_gene_order].to_numpy(copy=True)),
                    tissue_labels=torch.tensor([self.label_to_idx.get(str(label).strip().lower(), -1)
                                                for label in section['labels']], dtype=torch.long))

    def reference_frame(self):
        """Only call this on the training dataset; preserve all reference genes."""
        frames = []
        for name in self.names:
            frame = self.sections[name]['expression'][self.full_gene_order].copy()
            frame.insert(0, 'section', name)
            frame.insert(0, 'spot_id', [f'{name}:{spot}' for spot in frame.index])
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def build_datasets(data_root='data', dataset='her2st', fold=0, train_sections=None,
                   validation_sections=None, test_sections=None, seed=42, manifest=None,
                   core_genes=None, full_gene_order=None, label_to_idx=None):
    records = discover_sections(data_root, dataset, manifest)
    splits = split_sections(records, fold, train_sections, validation_sections, test_sections, seed=seed)
    # Fit the complete gene space, selection and ontology using training sections only.
    training = {name: load_section(records[name]) for name in splits['train']}
    if full_gene_order is None:
        full_gene_order = sorted(set.intersection(
            *(set(row['expression'].columns) for row in training.values())))
    if core_genes is None:
        combined = pd.concat([row['expression'][full_gene_order] for row in training.values()],
                             ignore_index=True)
        variance = combined.var(axis=0).sort_values(ascending=False, kind='stable')
        eligible = variance[variance > 0].index.tolist()
        if len(eligible) < 785:
            raise ValueError('Fewer than 785 variable genes in the training data.')
        core_genes = eligible[:785]
    if len(core_genes) != 785 or len(set(core_genes)) != 785:
        raise ValueError('Exactly 785 distinct directly supervised genes are required.')
    if not set(core_genes).issubset(full_gene_order):
        raise ValueError('Core genes must belong to the full training reference gene space.')
    if label_to_idx is None:
        if dataset.lower() == 'her2st':
            label_to_idx = dict(LABEL2ID)
        else:
            classes = sorted({str(label).strip().lower() for row in training.values()
                              for label in row['labels']
                              if str(label).strip().lower() not in ('undetermined', 'unknown', '')})
            label_to_idx = {label: i for i, label in enumerate(classes)}
    datasets = {key: SpatialExpressionDataset(names, records, core_genes, full_gene_order,
                                               label_to_idx, prepared=training)
                for key, names in splits.items()}
    return datasets, splits

