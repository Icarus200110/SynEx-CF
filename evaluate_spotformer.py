"""Evaluate a strict classifier checkpoint on held-out annotated sections."""
import argparse
import json
from pathlib import Path
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import discover_sections
from inference_spotformer import SimpleSpotInference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/spotformer_config.yaml')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_dir', default='evaluation_spot')
    parser.add_argument('--samples', nargs='+')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    data = config['data']
    inference = SimpleSpotInference(args.model)
    samples = args.samples or data.get('test_samples')
    if not samples:
        records = discover_sections(data['data_root'], data.get('dataset', 'her2st'),
                                     data.get('manifest'))
        samples = [sorted(records)[data.get('fold', 0)]]
    provenance = inference.checkpoint['section_splits']
    used = set(provenance['train']) | set(provenance['validation'])
    if used & set(samples):
        raise ValueError('Evaluation sections were used to fit/select the classifier.')
    records = discover_sections(data['data_root'], data.get('dataset', 'her2st'),
                                 data.get('manifest'))
    results = inference.predict_dataset(data['data_root'], samples, args.output_dir, records=records)
    labeled = results.loc[results['true_id'] >= 0]
    if not len(labeled):
        raise ValueError('No annotated spots in the held-out sections.')
    classes = list(range(len(inference.label_to_idx)))
    metrics = dict(accuracy=float(accuracy_score(labeled['true_id'], labeled['predicted_id'])),
                   macro_f1=float(f1_score(labeled['true_id'], labeled['predicted_id'],
                                          labels=classes, average='macro', zero_division=0)),
                   labeled_spots=len(labeled), sections=samples,
                   confusion_matrix=confusion_matrix(labeled['true_id'], labeled['predicted_id'],
                                                       labels=classes).tolist(),
                   label_to_idx=inference.label_to_idx)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

