"""Prediction helpers for manuscript-aligned checkpoints and section datasets."""
import torch
import numpy as np
from dataset import build_datasets
from SynEx_CF import SynEx_CF
from src.semantic import new_classifier
from src.contracts import IMPLEMENTATION_VERSION
from src.evaluation import gene_wise_pcc


def pk_load(fold, mode, flatten=False, data='her2st', neighs=8, prune='Grid',
            train_samples=None, *, validation_sections=None, test_sections=None,
            data_root='data', manifest=None):
    if flatten or neighs != 8 or prune != 'Grid':
        raise ValueError('The manuscript uses 224x224 RGB images and a physical 3x3 grid.')
    datasets, _ = build_datasets(data_root, data, fold, train_samples,
                                  validation_sections, test_sections, manifest=manifest)
    key = {'train': 'train', 'val': 'validation', 'validation': 'validation', 'test': 'test'}[mode]
    return datasets[key]


def load_expression_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Old checkpoints require retraining after the manuscript alignment.')
    semantic = None
    if checkpoint['model_config']['use_tissue']:
        semantic = new_classifier(checkpoint['label_to_idx'], checkpoint['semantic_model_config'])
    config = dict(checkpoint['model_config'])
    config['pretrained'] = False
    model = SynEx_CF(semantic_classifier=semantic, **config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def model_predict(model, dataset, device='cpu'):
    predictions, truths, centers = [], [], []
    model.eval()
    for section in dataset:
        prediction = model(section['patches'].to(device), section['positions'].to(device))
        predictions.append(prediction.float().cpu().numpy())
        truths.append(section['expression'].numpy())
        centers.append(section['centers'].numpy())
    return predictions, truths, centers


def get_R(data1, data2, dim=1, func=None):
    if dim != 1 or func is not None:
        raise ValueError('Use gene-wise PCC across spots within each evaluated section.')
    left = data1.X if hasattr(data1, 'X') else data1
    right = data2.X if hasattr(data2, 'X') else data2
    if hasattr(left, 'toarray'):
        left = left.toarray()
    if hasattr(right, 'toarray'):
        right = right.toarray()
    values = gene_wise_pcc(left, right)
    return values, np.full(values.shape, np.nan)

