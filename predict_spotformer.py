"""Single-section interface to the fold-specific tissue classifier."""
import argparse
import yaml
from inference_spotformer import SimpleSpotInference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/spotformer_config.yaml')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output_dir', default='predictions')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    inference = SimpleSpotInference(args.model_path)
    inference.predict_dataset(config['data']['data_root'], [args.sample], args.output_dir)


if __name__ == '__main__':
    main()

