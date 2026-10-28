"""Train a fold-specific ViT using only the expression training sections."""
import argparse
from pathlib import Path
import torch
import yaml
from dataset import build_datasets
from SynEX_CF_train import seed_everything
from src.semantic import train_classifier


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/spotformer_config.yaml')
    parser.add_argument('--output', default='outputs/tissue_classifier/best.pth')
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config['data']
    seed = config.get('seed', 42)
    seed_everything(seed)
    datasets, _ = build_datasets(
        data.get('data_root', 'data'), data.get('dataset', 'her2st'),
        data.get('fold', 0) if args.fold is None else args.fold,
        data.get('train_samples'), data.get('val_samples'), data.get('test_samples'),
        seed=seed, manifest=data.get('manifest'))
    # Classifier selection uses an inner split strictly inside expression training sections.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_classifier(datasets['train'], Path(args.output), device,
                     epochs=config.get('training', {}).get('epochs', 100),
                     lr=config.get('optimizer', {}).get('learning_rate', 5e-4),
                     seed=seed, model_type=config.get('model', {}).get('type', 'spotvit_small'),
                     chunk_size=config.get('training', {}).get('spot_chunk_size', 128))


if __name__ == '__main__':
    main()

