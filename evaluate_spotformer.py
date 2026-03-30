import os
import sys
import argparse
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.spotformer import SpotResNet, create_spotresnet
from src.data.spotformer_dataset import create_dataloaders

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SpotResNetLoss(nn.Module):
    """Loss function for SpotResNet"""
    
    def __init__(self, use_focal=True, focal_alpha=1.0, focal_gamma=2.0, class_weights=None):
        super().__init__()
        
        if use_focal:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, spot_logits, spot_labels, mask):
        """
        Args:
            spot_logits: [batch_size, max_spots, num_classes]
            spot_labels: [batch_size, max_spots]
            mask: [batch_size, max_spots] - 有效spot的mask
        """
        # Spot级别损失
        valid_spot_logits = spot_logits[mask]
        valid_spot_labels = spot_labels[mask]
        loss = self.criterion(valid_spot_logits, valid_spot_labels)
        
        return loss

class SpotResNetEvaluator:
    """SpotResNet evaluator for evaluating model performance and generating visualization results"""
    
    def __init__(self, config_path, model_path):
        """
        Initialize evaluator
        
        Args:
            config_path: Configuration file path
            model_path: Model weight path
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up logging
        self.setup_logging()
        
        # Create output directories
        self.setup_directories()
        
        # Load data
        self.setup_data()
        
        # Load model
        self.setup_model(model_path)
        
        # Initialize result storage
        self.results = {}
        
        # Set up loss function, consistent with training
        self.setup_loss()
        
        self.logger.info("Evaluator initialization completed")
    
    def setup_logging(self):
        """Set up logging"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.config['paths']['output_dir']) / f"evaluation_{timestamp}"
        self.visualization_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.output_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Evaluation result output directory: {self.output_dir}")
    
    def setup_data(self):
        """Set up data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # Get configuration parameters
        auto_patch_size = self.config['data'].get('auto_patch_size', False)
        skip_undetermined = self.config['data'].get('skip_undetermined', True)
        
        # Disable undersampling during evaluation, use original data distribution
        undersample_majority = False
        undersample_ratio = self.config['data'].get('undersample_ratio', 0.5)
        
        # Decide whether to pass patch_size parameter based on auto_patch_size
        create_dataloaders_kwargs = {
            'data_root': self.config['data']['data_root'],
            'train_samples': self.config['data']['train_samples'],
            'test_samples': self.config['data'].get('test_samples'),
            'batch_size': self.config['inference']['batch_size'],
            'max_spots_per_sample': self.config['data']['max_spots_per_sample'],
            'num_workers': self.config['data']['num_workers'],
            'auto_patch_size': auto_patch_size,
            'skip_undetermined': skip_undetermined,
            'undersample_majority': undersample_majority,
            'undersample_ratio': undersample_ratio
        }
        
        # Only pass patch_size when auto_patch_size is False
        if not auto_patch_size:
            create_dataloaders_kwargs['patch_size'] = self.config['model']['patch_size']
        
        # Create data loaders
        self.dataloaders, self.label_to_idx, self.idx_to_label = create_dataloaders(**create_dataloaders_kwargs)
        
        self.num_classes = len(self.label_to_idx)
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Label mapping: {self.label_to_idx}")
    
    def setup_model(self, model_path):
        """Set up model"""
        self.logger.info("Setting up model...")
        
        # Create standard model
        self.logger.info("Creating standard model...")
        self.model = create_spotresnet(
            num_classes=self.num_classes,
            model_type=self.config['model']['type'],
            patch_size=self.config['model']['patch_size'],
            use_spatial_attention=self.config['model'].get('use_spatial_attention', True),
            use_fpn=self.config['model'].get('use_fpn', True),
            fpn_out_channels=self.config['model'].get('fpn_out_channels', 256)
        )
        self.use_correction = False
        
        # Load model weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # Use strict=False to allow partial loading, handling model architecture mismatch
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            self.logger.info(f"Successfully loaded model weights: {model_path}")
        else:
            raise FileNotFoundError(f"Model weight file does not exist: {model_path}")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def setup_loss(self):
        """Set up loss function, consistent with training"""
        loss_config = self.config['loss']
        
        # Get class weights
        class_weights = loss_config.get('class_weights', None)
        if class_weights is not None:
            # Ensure class weights match the number of classes
            if len(class_weights) != self.num_classes:
                self.logger.warning(f"Number of class weights({len(class_weights)}) does not match number of classes({self.num_classes}), will not use class weights")
                class_weights = None
            else:
                # Convert to tensor and move to device
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        self.criterion = SpotResNetLoss(
            use_focal=loss_config.get('use_focal', False),
            focal_alpha=loss_config.get('focal_alpha', 0.25),  # Use focal_alpha value from config file
            focal_gamma=loss_config.get('focal_gamma', 3.0),   # Use focal_gamma value from config file
            class_weights=class_weights  # Add class weights parameter
        )
        
        self.logger.info("Evaluator initialization completed")
    
    def evaluate(self):
        """Evaluate model performance"""
        self.logger.info("Starting model evaluation...")
        
        # Get test data loader
        test_loader = self.dataloaders.get('test')
        if test_loader is None:
            self.logger.warning("Test dataset not found, using validation dataset for evaluation")
            test_loader = self.dataloaders.get('val')
        
        if test_loader is None:
            raise ValueError("Test or validation dataset not found")
        
        # Initialize variables
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0
        total_samples = 0
        
        # Store prediction results and true labels for each sample for spatial visualization
        self.sample_predictions = {}  # {sample_id: {'predictions': [], 'labels': [], 'positions': []}}
        
        # Use the same loss function as during training
        criterion = self.criterion
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Get data
                patches = batch['patches'].to(self.device)
                positions = batch['positions'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['mask'].to(self.device)
                sample_ids = batch['sample_ids']
                
                # Forward propagation - consistent with training process, pass mask parameter
                # For standard model
                spot_logits = self.model(patches, positions, mask)
                
                # Calculate loss
                valid_spot_logits = spot_logits[mask]
                valid_spot_labels = labels[mask]
                # Create a mask corresponding to valid_spot_logits, all True
                valid_mask = torch.ones_like(valid_spot_labels, dtype=torch.bool)
                loss = criterion(valid_spot_logits, valid_spot_labels, valid_mask)
                total_loss += loss.item() * valid_spot_logits.size(0)
                total_samples += valid_spot_logits.size(0)
                
                # Get prediction results
                probabilities = torch.softmax(spot_logits, dim=2)  # Calculate softmax on class dimension
                predictions = torch.argmax(probabilities, dim=2)   # Get prediction results on class dimension
                
                # Collect results
                # Directly use mask to filter valid predictions, no need to flatten
                valid_predictions = predictions[mask]
                valid_probabilities = probabilities[mask]
                
                all_predictions.extend(valid_predictions.cpu().numpy())
                all_labels.extend(valid_spot_labels.cpu().numpy())
                all_probabilities.extend(valid_probabilities.cpu().numpy())
                
                # Collect prediction results and true labels for each sample for spatial visualization
                for i, sample_id in enumerate(sample_ids):
                    if sample_id not in self.sample_predictions:
                        self.sample_predictions[sample_id] = {
                            'predictions': [],
                            'labels': [],
                            'positions': [],
                            'mask': []
                        }
                    
                    # Get valid spots for current sample
                    # Assume mask shape is [batch_size, num_spots]
                    sample_mask = mask[i]
                    valid_indices = torch.where(sample_mask)[0]
                    
                    # Add prediction results, true labels and positions
                    # Directly use mask to filter results
                    for idx in valid_indices:
                        self.sample_predictions[sample_id]['predictions'].append(predictions[i][idx].item())
                        self.sample_predictions[sample_id]['labels'].append(labels[i][idx].item())
                        self.sample_predictions[sample_id]['positions'].append(positions[i][idx].cpu().numpy())
                        self.sample_predictions[sample_id]['mask'].append(True)
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Get actual labels
        unique_labels = sorted(set(all_labels))
        
        # Calculate precision, recall, F1 score
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, labels=unique_labels, average=None, zero_division=0
        )
        
        # Generate classification report
        target_names = [self.idx_to_label[i] for i in unique_labels]
        
        class_report = classification_report(
            all_labels, all_predictions, 
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities)
        }
        
        # Print results
        self.logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f}, Average loss: {avg_loss:.4f}")
        self.logger.info("\nClassification report:\n" + class_report)
        
        return self.results
    
    def save_results(self):
        """Save evaluation results"""
        self.logger.info("Saving evaluation results...")
        
        # Save numerical results
        # Get actual labels
        unique_labels = sorted(set(self.results['labels']))
        target_names = [self.idx_to_label[i] for i in unique_labels]
        
        results_df = pd.DataFrame({
            'Class': target_names,
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1-Score': self.results['f1'],
            'Support': self.results['support']
        })
        
        results_path = self.output_dir / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Evaluation results saved to: {results_path}")
        
        # Save prediction results
        predictions_df = pd.DataFrame({
            'True_Label': [self.idx_to_label[label] for label in self.results['labels']],
            'Predicted_Label': [self.idx_to_label[pred] for pred in self.results['predictions']],
            'True_Index': self.results['labels'],
            'Predicted_Index': self.results['predictions']
        })
        
        # Add probability for each class
        for i in unique_labels:
            predictions_df[f'Prob_{self.idx_to_label[i]}'] = self.results['probabilities'][:, i]
        
        predictions_path = self.output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Prediction results saved to: {predictions_path}")
        
        # Save classification report
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.results['classification_report'])
        self.logger.info(f"Classification report saved to: {report_path}")
    
    def visualize_results(self):
        """Visualize evaluation results"""
        self.logger.info("Generating visualization results...")
        
        # 1. Plot confusion matrix
        self.plot_confusion_matrix()
        
        # 2. Plot class performance metrics
        self.plot_class_metrics()
        
        # 3. Plot prediction probability distribution
        self.plot_probability_distribution()
        
        # 4. Plot error analysis
        self.plot_error_analysis()
        
        # 5. Plot comparison visualization between true spots and predicted spots
        self.plot_spot_comparison(num_samples=10)
        
        self.logger.info(f"Visualization results saved to: {self.visualization_dir}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = self.results['confusion_matrix']
        # Get actual labels
        unique_labels = sorted(set(self.results['labels']))
        class_names = [self.idx_to_label[i] for i in unique_labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save image
        cm_path = self.visualization_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save image
        cm_norm_path = self.visualization_dir / "confusion_matrix_normalized.png"
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_metrics(self):
        """Plot class performance metrics"""
        # Get actual labels
        unique_labels = sorted(set(self.results['labels']))
        
        metrics_df = pd.DataFrame({
            'Class': [self.idx_to_label[i] for i in unique_labels],
            'Precision': [self.results['precision'][i] for i in range(len(unique_labels)) if i < len(self.results['precision'])],
            'Recall': [self.results['recall'][i] for i in range(len(unique_labels)) if i < len(self.results['recall'])],
            'F1-Score': [self.results['f1'][i] for i in range(len(unique_labels)) if i < len(self.results['f1'])]
        })
        
        # Convert data to long format for plotting
        metrics_melted = pd.melt(metrics_df, id_vars=['Class'], 
                                value_vars=['Precision', 'Recall', 'F1-Score'],
                                var_name='Metric', value_name='Value')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_melted)
        plt.title('Performance Metrics by Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Metric')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save image
        metrics_path = self.visualization_dir / "class_metrics.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_probability_distribution(self):
        """Plot prediction probability distribution"""
        probabilities = self.results['probabilities']
        predictions = self.results['predictions']
        labels = self.results['labels']
        
        # Get actual labels
        unique_labels = sorted(set(labels))
        
        # Plot probability distribution for each class
        for class_idx in unique_labels:
            class_name = self.idx_to_label[class_idx]
            
            # Get probabilities for correct and incorrect predictions
            correct_mask = (predictions == labels) & (labels == class_idx)
            incorrect_mask = (predictions != labels) & (labels == class_idx)
            
            correct_probs = probabilities[correct_mask, class_idx] if np.any(correct_mask) else np.array([])
            incorrect_probs = probabilities[incorrect_mask, class_idx] if np.any(incorrect_mask) else np.array([])
            
            plt.figure(figsize=(10, 6))
            
            if len(correct_probs) > 0:
                sns.kdeplot(correct_probs, label='Correct Predictions', shade=True)
            
            if len(incorrect_probs) > 0:
                sns.kdeplot(incorrect_probs, label='Incorrect Predictions', shade=True)
            
            plt.title(f'Prediction Probability Distribution for Class "{class_name}"')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            
            # Save image
            prob_path = self.visualization_dir / f"probability_distribution_{class_name.replace(' ', '_')}.png"
            plt.savefig(prob_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_spot_comparison(self, num_samples=10):
        """
        Generate comparison visualization between true spots and predicted spots
        for each sample.
        
        Args:
            num_samples: Number of samples to visualize
        """
        self.logger.info("Generating comparison visualization between true spots and predicted spots...")
        
        # Ensure results directory exists
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Get sample ID list
        sample_ids = list(self.sample_predictions.keys())
        
        # Limit the number of samples for visualization
        if len(sample_ids) > num_samples:
            sample_ids = sample_ids[:num_samples]
        
        # Generate comparison visualization for each sample
        for sample_id in sample_ids:
            sample_data = self.sample_predictions[sample_id]
            predictions = np.array(sample_data['predictions'])
            labels = np.array(sample_data['labels'])
            positions = np.array(sample_data['positions'])
            
            # Create comparison visualization image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot true spots
            ax1.set_title(f"True Spot (Sample ID: {sample_id})", fontsize=14)
            self._plot_spots(ax1, positions, labels, "True")
            
            # Plot predicted spots
            ax2.set_title(f"Predicted Spot (Sample ID: {sample_id})", fontsize=14)
            self._plot_spots(ax2, positions, predictions, "Predicted")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save image
            output_path = os.path.join(self.visualization_dir, f"spot_comparison_{sample_id}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved spot comparison visualization for sample {sample_id} to {output_path}")
        
        self.logger.info(f"Visualization results saved to: {self.visualization_dir}")
    
    def _plot_spots(self, ax, positions, labels, title_prefix):
        """
        Plot spots on the given axis
        
        Args:
            ax: matplotlib axis object
            positions: position coordinates of spots
            labels: labels of spots
            title_prefix: title prefix ("True" or "Predicted")
        """
        # Define color mapping
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.idx_to_label)))
        
        # Plot spots for each class
        for class_idx in self.idx_to_label:
            class_name = self.idx_to_label[class_idx]
            mask = labels == class_idx
            
            if np.any(mask):
                # Get positions for current class
                class_positions = positions[mask]
                
                # Plot spots
                ax.scatter(
                    class_positions[:, 0], 
                    class_positions[:, 1], 
                    c=[colors[class_idx]], 
                    label=class_name,
                    alpha=0.7,
                    s=50,
                    marker='o'
                )
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)
        
        # Set axis labels
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        
        # Set title
        ax.set_title(f"{title_prefix} Spot Distribution", fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Automatically adjust axis range
        if len(positions) > 0:
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            
            # Add some margin
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05
            
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    def plot_error_analysis(self):
        """Plot error analysis"""
        predictions = self.results['predictions']
        labels = self.results['labels']
        probabilities = self.results['probabilities']
        
        # Find incorrectly predicted samples
        error_mask = predictions != labels
        error_indices = np.where(error_mask)[0]
        
        if len(error_indices) == 0:
            self.logger.info("No incorrectly predicted samples, skipping error analysis")
            return
        
        # Get incorrectly predicted samples
        error_labels = labels[error_mask]
        error_predictions = predictions[error_mask]
        error_probabilities = probabilities[error_mask]
        
        # Get actual labels
        unique_labels = sorted(set(labels))
        
        # Analyze error types
        error_types = {}
        for true_label, pred_label in zip(error_labels, error_predictions):
            error_type = f"{self.idx_to_label[true_label]} → {self.idx_to_label[pred_label]}"
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        # Plot error type distribution
        plt.figure(figsize=(12, 6))
        error_types_df = pd.DataFrame(list(error_types.items()), columns=['Error Type', 'Count'])
        error_types_df = error_types_df.sort_values('Count', ascending=False)
        
        sns.barplot(x='Error Type', y='Count', data=error_types_df)
        plt.title('Error Type Distribution')
        plt.xlabel('Error Type (True → Predicted)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save image
        error_path = self.visualization_dir / "error_analysis.png"
        plt.savefig(error_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze probability distribution of incorrect predictions
        plt.figure(figsize=(10, 6))
        
        # Get probabilities of predicted classes for incorrectly predicted samples
        pred_class_probs = error_probabilities[range(len(error_predictions)), error_predictions]
        
        sns.histplot(pred_class_probs, bins=20, kde=True)
        plt.title('Probability Distribution of Incorrect Predictions')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save image
        error_prob_path = self.visualization_dir / "error_probability_distribution.png"
        plt.savefig(error_prob_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate SpotResNet model')
    parser.add_argument('--config', type=str, default='./configs/spotformer_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=False, default='./outputs/spotresnet_20250917_180819/checkpoints/best.pth',
                       help='Model weight path')
    parser.add_argument('--output_dir', type=str, default='evaluation_spot',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SpotResNetEvaluator(args.config, args.model)
    
    # Evaluate model
    evaluator.evaluate()
    
    # Save results
    evaluator.save_results()
    
    # Visualize results
    evaluator.visualize_results()
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()