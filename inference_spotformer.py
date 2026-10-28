"""Predict tissue phenotypes from image patches, including unlabeled sections."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from src.semantic import load_classifier
from src.data.spotformer_dataset import SpotFormerDataset


class SimpleSpotInference:
    def __init__(self, model_path, config_path=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        saved = torch.load(model_path, map_location='cpu', weights_only=False)
        self.label_to_idx = saved['label_to_idx']
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}
        self.model, self.checkpoint = load_classifier(model_path, self.label_to_idx, device=self.device)

    @torch.no_grad()
    def predict_dataset(self, data_root, samples, output_dir=None, save_results=True,
                        visualize=False, records=None):
        dataset = SpotFormerDataset(data_root, samples, mode='inference',
                                    skip_undetermined=False, label_to_idx=self.label_to_idx,
                                    records=records)
        frames = []
        for section in dataset:
            probabilities = []
            for chunk in section['patches'].split(128):
                output = self.model(chunk.to(self.device)[None])[0]
                probabilities.append(output.float().softmax(-1).cpu())
            probabilities = torch.cat(probabilities).numpy()
            prediction = probabilities.argmax(-1)
            frame = pd.DataFrame(dict(section=section['sample_id'], spot_id=section['spot_ids'],
                                      x=section['spatial_coords'][:, 0].numpy(),
                                      y=section['spatial_coords'][:, 1].numpy(),
                                      pixel_x=section['positions'][:, 0].numpy(),
                                      pixel_y=section['positions'][:, 1].numpy(),
                                      predicted_id=prediction,
                                      predicted_class=[self.idx_to_label[int(i)] for i in prediction],
                                      confidence=probabilities.max(-1),
                                      true_id=section['labels'].numpy()))
            for label, i in self.label_to_idx.items():
                frame[f'probability_{label}'] = probabilities[:, i]
            frames.append(frame)
        results = pd.concat(frames, ignore_index=True)
        if output_dir is not None and save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            results.to_csv(Path(output_dir) / 'tissue_predictions.tsv', sep='\t', index=False)
        return results

    @staticmethod
    def get_prediction_summary(results):
        return dict(total_spots=len(results),
                    num_classes_predicted=int(results['predicted_class'].nunique()),
                    average_confidence=float(results['confidence'].mean()),
                    class_distribution=results['predicted_class'].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--samples', nargs='+', required=True)
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--no_save', action='store_true')
    parser.add_argument('--no_visualize', action='store_true')
    args = parser.parse_args()
    inference = SimpleSpotInference(args.model, args.config)
    results = inference.predict_dataset(args.data_root, args.samples, args.output_dir,
                                         save_results=not args.no_save)
    print(inference.get_prediction_summary(results))


if __name__ == '__main__':
    main()

