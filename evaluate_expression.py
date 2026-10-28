"""Aggregate existing fold-level gene PCC files and paired baseline comparisons."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.evaluation import summarize_sections, bootstrap_sections_genes, paired_wilcoxon_fdr


def load_fold_scores(paths):
    sections, experiment_ids = {}, set()
    for value in paths:
        path = Path(value).resolve()
        # protocol.json prevents combining main-comparison and ablation runs.
        protocol = json.loads((path.parent / 'protocol.json').read_text(encoding='utf-8'))
        experiment_ids.add(protocol['experiment'])
        if len(experiment_ids) > 1:
            raise ValueError('Main comparison and ablation results must be summarized separately.')
        frame = pd.read_csv(path, sep='\t', index_col=0)
        if frame.index.duplicated().any():
            raise ValueError(f'Duplicate gene identifiers in {path}.')
        expected = set(protocol['section_splits']['test'])
        if set(frame.columns) != expected:
            raise ValueError(f'PCC section columns disagree with the held-out split: {path}')
        for section in frame:
            if section in sections:
                raise ValueError(f'Repeated evaluated section {section}; do not pool repeated training runs.')
            sections[section] = frame[section]
    return sections, experiment_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', nargs='+', required=True,
                        help='Gene-indexed full_gene_pcc.tsv files from separate held-out folds.')
    parser.add_argument('--baseline', nargs='+', action='append', default=[],
                        metavar=('NAME', 'SCORES'),
                        help='Repeat with a model name followed by its paired fold PCC TSV files.')
    parser.add_argument('--output', default='aggregate_metrics.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    model, experiment = load_fold_scores(args.scores)
    summary = summarize_sections(model)
    summary.update(bootstrap_sections_genes(model, seed=args.seed))
    summary['experiment'] = next(iter(experiment))
    baselines = {}
    for item in args.baseline:
        if len(item) < 2 or item[0] in baselines:
            raise ValueError('Each baseline requires a unique model name and at least one PCC file.')
        scores, baseline_experiment = load_fold_scores(item[1:])
        if baseline_experiment != experiment or set(scores) != set(model):
            raise ValueError('Baseline comparisons require the same experiment and held-out sections.')
        baselines[item[0]] = scores
    if baselines:
        summary['paired_comparisons'] = paired_wilcoxon_fdr(model, baselines)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

