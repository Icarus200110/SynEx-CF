"""Gene-wise evaluation within sections; undefined correlations remain excluded."""
import numpy as np
from scipy.stats import wilcoxon


def gene_wise_pcc(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError('Expected matching [spots, genes] matrices.')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions or measured values cannot be scored.')
    x, y = truth - truth.mean(0), prediction - prediction.mean(0)
    denominator = np.sqrt((x * x).sum(0) * (y * y).sum(0))
    result = np.full(truth.shape[1], np.nan)
    valid = denominator > 1e-12
    result[valid] = np.clip((x[:, valid] * y[:, valid]).sum(0) / denominator[valid], -1, 1)
    return result


def summarize_sections(section_pcc):
    vectors = list(section_pcc.values()) if isinstance(section_pcc, dict) else list(section_pcc)
    finite = [np.asarray(v)[np.isfinite(v)] for v in vectors]
    values = np.concatenate(finite) if finite else np.empty(0)
    if not len(values):
        raise ValueError('No defined gene-wise PCC values.')
    return dict(mean_pcc=float(values.mean()), median_pcc=float(np.median(values)),
                evaluated_gene_section_pairs=int(len(values)), sections=len(vectors))


def bootstrap_sections_genes(section_pcc, repetitions=2000, seed=42, confidence=0.95):
    """Resample sections, then genes within sampled sections; exclude undefined genes."""
    vectors = [np.asarray(v)[np.isfinite(v)] for v in section_pcc.values()]
    vectors = [v for v in vectors if len(v)]
    if not vectors or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError('Require defined correlations, positive repetitions and 0 < confidence < 1.')
    rng, statistics = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sampled = [rng.choice(vectors[i], len(vectors[i]), replace=True)
                   for i in rng.integers(0, len(vectors), size=len(vectors))]
        values = np.concatenate(sampled)
        statistics.append((values.mean(), np.median(values)))
    alpha = (1 - confidence) / 2
    intervals = np.quantile(statistics, [alpha, 1 - alpha], axis=0)
    return dict(mean_pcc_ci=intervals[:, 0].tolist(), median_pcc_ci=intervals[:, 1].tolist(),
                confidence=confidence, bootstrap_repetitions=repetitions)


def paired_wilcoxon_fdr(model, baselines):
    """Inputs: section -> gene-indexed pandas Series. Pair genes before section means."""
    comparisons, pvalues = {}, []
    for name, baseline in baselines.items():
        differences = []
        for section in sorted(set(model) & set(baseline)):
            paired = model[section].to_frame('model').join(
                baseline[section].rename('baseline'), how='inner')
            paired = paired.loc[np.isfinite(paired).all(axis=1)]
            if len(paired):
                differences.append(float((paired['model'] - paired['baseline']).mean()))
        if not differences:
            raise ValueError(f'No paired section/gene results for {name}.')
        differences = np.asarray(differences)
        pvalue = (1.0 if np.all(differences == 0) else
                  float(wilcoxon(differences, alternative='two-sided').pvalue))
        comparisons[name] = dict(p_value=pvalue, paired_sections=len(differences))
        pvalues.append(pvalue)
    if pvalues:
        order = np.argsort(pvalues)
        ranked = np.asarray(pvalues)[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
        adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
        for rank, original in enumerate(order):
            comparisons[list(comparisons)[original]]['fdr_q_value'] = float(adjusted[rank])
    return comparisons

