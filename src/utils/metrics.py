"""Continuous expression metrics, computed within each complete tissue section."""
import numpy as np
import torch
from scipy.stats import spearmanr
from src.evaluation import gene_wise_pcc, summarize_sections


def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_correlation(y_true, y_pred):
    values = gene_wise_pcc(y_true, y_pred)
    valid = values[np.isfinite(values)]
    return (float(valid.mean()) if len(valid) else float('nan')), values


def spearman_correlation(y_true, y_pred):
    x, y = np.asarray(y_true), np.asarray(y_pred)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Expected matching finite [spots, genes] expression matrices.')
    values = np.full(x.shape[1], np.nan)
    for gene in range(x.shape[1]):
        if np.ptp(x[:, gene]) > 0 and np.ptp(y[:, gene]) > 0:
            values[gene] = spearmanr(x[:, gene], y[:, gene]).statistic
    valid = values[np.isfinite(values)]
    return (float(valid.mean()) if len(valid) else float('nan')), values


def cosine_similarity_metric(y_true, y_pred):
    x, y = np.asarray(y_true), np.asarray(y_pred)
    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    values = np.divide((x * y).sum(0), denominator,
                       out=np.full(x.shape[1], np.nan), where=denominator > 1e-12)
    valid = values[np.isfinite(values)]
    return (float(valid.mean()) if len(valid) else float('nan')), values


def explained_variance_score(y_true, y_pred):
    x, y = np.asarray(y_true), np.asarray(y_pred)
    variance = np.var(x, axis=0)
    values = np.full(x.shape[1], np.nan)
    valid = variance > 1e-12
    values[valid] = 1 - np.var(x - y, axis=0)[valid] / variance[valid]
    finite = values[np.isfinite(values)]
    return (float(finite.mean()) if len(finite) else float('nan')), values


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Accept section dictionaries; never pool spots across sections or round log expression."""
    model.eval()
    correlations, results = {}, {}
    for index, section in enumerate(dataloader):
        patches = section['patches'].to(device)
        positions = section['positions'].to(device)
        prediction = model(patches, positions).float().cpu().numpy()
        truth = section['expression'].numpy()
        if prediction.ndim == 3:
            if len(prediction) != 1:
                raise ValueError('Each evaluation batch must contain exactly one section.')
            prediction, truth = prediction[0], truth[0]
        name = section.get('section', str(index))
        name = name[0] if isinstance(name, (tuple, list)) else name
        correlations[name] = gene_wise_pcc(truth, prediction)
        results[name] = dict(mse=mean_squared_error(truth, prediction),
                             gene_pcc=correlations[name].tolist())
    summary = summarize_sections(correlations)
    summary['section_metrics'] = results
    return summary

