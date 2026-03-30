import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.spotformer import SpotResNet, create_spotresnet
from src.data.spotformer_dataset import SpotFormerDataset

class SimpleSpotInference:
    """简单spot分类器的推理器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径（可选）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型检查点
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从检查点中获取配置，或从文件加载
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.checkpoint.get('config', {})
        
        # 获取标签映射
        self.label_mapping = self.checkpoint.get('label_mapping', {})
        self.label_to_idx = self.label_mapping.get('label_to_idx', {})
        self.idx_to_label = self.label_mapping.get('idx_to_label', {})
        self.num_classes = len(self.idx_to_label)
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载权重
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # 检查模型类型
        model_type = self.checkpoint.get('model_type', 'Standard')
        use_correction = False  # 轻量化ViT模型不支持空间矫正
        
        logging.info(f"Model loaded from {model_path}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Number of classes: {self.num_classes}")
        logging.info(f"Classes: {list(self.idx_to_label.values())}")
    
    def _create_model(self):
        """创建模型"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'spotvit_small')
        
        # 支持轻量化ViT模型
        model = create_spotresnet(
            num_classes=self.num_classes,
            model_type=model_type,
            patch_size=model_config.get('patch_size', 64)
        )
        logging.info(f"Using model: {model_type}")
        
        # 加载预训练权重（如果有）
        pretrained_path = model_config.get('pretrained_path')
        if pretrained_path and os.path.exists(pretrained_path):
            logging.info(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def predict_dataset(self, data_root: str, samples: list, output_dir: str = None,
                       save_results: bool = True, visualize: bool = None) -> dict:
        """
        对数据集进行预测
        
        Args:
            data_root: 数据根目录
            samples: 样本列表
            output_dir: 输出目录
            save_results: 是否保存结果
            visualize: 是否可视化（如果为None，则从配置文件中读取）
            
        Returns:
            预测结果字典
        """
        # 从配置文件中获取推理设置
        inference_config = self.config.get('inference', {})
        batch_size = inference_config.get('batch_size', 64)
        output_probabilities = inference_config.get('output_probabilities', True)
        save_features = inference_config.get('save_features', False)
        
        # 如果未指定visualize参数，则从配置文件中读取
        if visualize is None:
            visualize = inference_config.get('visualization', True)
        
        # 创建数据集
        dataset = SpotFormerDataset(
            data_root=data_root,
            samples=samples,
            patch_size=self.config.get('data', {}).get('patch_size', 64),
            max_spots_per_sample=self.config.get('data', {}).get('max_spots_per_sample', 1000),
            mode='test',
            auto_patch_size=self.config.get('data', {}).get('auto_patch_size', False),
            skip_undetermined=False  # 推理时不跳过任何样本
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=SpotFormerDataset.collate_fn,
            pin_memory=True
        )
        
        # 预测
        all_predictions = []
        all_probabilities = []
        all_sample_names = []
        all_spot_indices = []
        all_coordinates = []
        
        logging.info(f"Starting inference on {len(dataset)} spots...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                patches = batch['patches'].to(self.device)
                positions = batch['positions'].to(self.device)
                mask = batch['mask'].to(self.device)
                sample_names = batch['sample_names']
                spot_indices = batch['spot_indices']
                coordinates = batch.get('coordinates', [])
                
                # 前向传播
                # 轻量化ViT模型不支持空间矫正，直接调用模型
                outputs = self.model(patches, positions, mask)  # [batch_size, num_spots, num_classes]
                
                probabilities = F.softmax(outputs, dim=-1)
                predictions = torch.argmax(outputs, dim=-1)
                
                # 收集结果
                for i in range(len(sample_names)):
                    valid_mask = mask[i]
                    valid_preds = predictions[i][valid_mask].cpu().numpy()
                    valid_probs = probabilities[i][valid_mask].cpu().numpy()
                    valid_sample_names = [sample_names[i]] * len(valid_preds)
                    valid_spot_indices = spot_indices[i][:len(valid_preds)]
                    
                    all_predictions.extend(valid_preds)
                    all_probabilities.extend(valid_probs)
                    all_sample_names.extend(valid_sample_names)
                    all_spot_indices.extend(valid_spot_indices)
                    
                    if coordinates:
                        valid_coords = coordinates[i][:len(valid_preds)]
                        all_coordinates.extend(valid_coords)
        
        # 整理结果
        results = {
            'sample_names': all_sample_names,
            'spot_indices': all_spot_indices,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'predicted_labels': [self.idx_to_label[pred] for pred in all_predictions]
        }
        
        if all_coordinates:
            results['coordinates'] = all_coordinates
        
        logging.info(f"Inference completed. Processed {len(all_predictions)} spots.")
        
        # 保存和可视化结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            if save_results:
                self._save_results(results, output_dir)
            
            if visualize:
                self._visualize_results(results, output_dir)
        
        return results
    
    def predict_single_sample(self, data_root: str, sample_name: str, 
                             output_dir: str = None) -> dict:
        """
        对单个样本进行预测
        
        Args:
            data_root: 数据根目录
            sample_name: 样本名称
            output_dir: 输出目录
            
        Returns:
            预测结果字典
        """
        return self.predict_dataset(
            data_root=data_root,
            samples=[sample_name],
            output_dir=output_dir
        )
    
    def _save_results(self, results: dict, output_dir: str):
        """保存预测结果"""
        # 创建DataFrame
        df_data = {
            'sample_name': results['sample_names'],
            'spot_index': results['spot_indices'],
            'predicted_class': results['predicted_labels'],
            'prediction_confidence': [probs[pred] for probs, pred in 
                                    zip(results['probabilities'], results['predictions'])]
        }
        
        # 添加坐标信息（如果有）
        if 'coordinates' in results:
            coords = results['coordinates']
            df_data['x_coordinate'] = [coord[0] for coord in coords]
            df_data['y_coordinate'] = [coord[1] for coord in coords]
        
        # 添加所有类别的概率
        for i, class_name in enumerate(self.idx_to_label.values()):
            df_data[f'prob_{class_name}'] = [probs[i] for probs in results['probabilities']]
        
        df = pd.DataFrame(df_data)
        
        # 保存CSV文件
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        
        # 保存JSON文件（包含更多详细信息）
        json_results = {
            'metadata': {
                'model_path': self.checkpoint.get('model_path', 'unknown'),
                'num_classes': self.num_classes,
                'class_names': list(self.idx_to_label.values()),
                'total_spots': len(results['predictions']),
                'inference_time': datetime.now().isoformat()
            },
            'predictions': results
        }
        
        json_path = os.path.join(output_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        logging.info(f"Detailed results saved to {json_path}")
    
    def _visualize_results(self, results: dict, output_dir: str):
        """可视化预测结果"""
        # 1. 类别分布饼图
        self._plot_class_distribution(results, output_dir)
        
        # 2. 置信度分布直方图
        self._plot_confidence_distribution(results, output_dir)
        
        # 3. 每个样本的类别分布
        self._plot_sample_class_distribution(results, output_dir)
        
        # 4. 空间分布图（如果有坐标信息）
        if 'coordinates' in results:
            self._plot_spatial_distribution(results, output_dir)
    
    def _plot_class_distribution(self, results: dict, output_dir: str):
        """绘制类别分布饼图"""
        predicted_labels = results['predicted_labels']
        class_counts = pd.Series(predicted_labels).value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        wedges, texts, autotexts = plt.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        plt.title('Predicted Class Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        # 添加图例
        plt.legend(wedges, [f'{label}: {count}' for label, count in class_counts.items()],
                  title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Class distribution plot saved")
    
    def _plot_confidence_distribution(self, results: dict, output_dir: str):
        """绘制置信度分布直方图"""
        confidences = [probs[pred] for probs, pred in 
                      zip(results['probabilities'], results['predictions'])]
        
        plt.figure(figsize=(12, 6))
        
        # 总体置信度分布
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # 按类别的置信度分布
        plt.subplot(1, 2, 2)
        for class_name in self.idx_to_label.values():
            class_confidences = [confidences[i] for i, label in enumerate(results['predicted_labels']) 
                               if label == class_name]
            if class_confidences:
                plt.hist(class_confidences, bins=20, alpha=0.6, label=class_name)
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Confidence distribution plot saved")
    
    def _plot_sample_class_distribution(self, results: dict, output_dir: str):
        """绘制每个样本的类别分布"""
        df = pd.DataFrame({
            'sample_name': results['sample_names'],
            'predicted_class': results['predicted_labels']
        })
        
        sample_class_counts = df.groupby(['sample_name', 'predicted_class']).size().unstack(fill_value=0)
        
        # 堆叠柱状图
        plt.figure(figsize=(12, 8))
        sample_class_counts.plot(kind='bar', stacked=True, colormap='Set3')
        plt.title('Class Distribution by Sample', fontsize=16, fontweight='bold')
        plt.xlabel('Sample Name')
        plt.ylabel('Number of Spots')
        plt.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Sample class distribution plot saved")
    
    def _plot_spatial_distribution(self, results: dict, output_dir: str):
        """绘制空间分布图"""
        if 'coordinates' not in results:
            return
        
        df = pd.DataFrame({
            'x': [coord[0] for coord in results['coordinates']],
            'y': [coord[1] for coord in results['coordinates']],
            'sample_name': results['sample_names'],
            'predicted_class': results['predicted_labels']
        })
        
        # 为每个样本创建空间分布图
        samples = df['sample_name'].unique()
        
        for sample in samples:
            sample_df = df[df['sample_name'] == sample]
            
            plt.figure(figsize=(12, 10))
            
            # 创建颜色映射
            unique_classes = sample_df['predicted_class'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            color_map = dict(zip(unique_classes, colors))
            
            # 绘制散点图
            for class_name in unique_classes:
                class_data = sample_df[sample_df['predicted_class'] == class_name]
                plt.scatter(class_data['x'], class_data['y'], 
                           c=[color_map[class_name]], label=class_name, 
                           alpha=0.7, s=20)
            
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'Spatial Distribution of Predicted Classes - {sample}', 
                     fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'spatial_distribution_{sample}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info("Spatial distribution plots saved")
    
    def get_prediction_summary(self, results: dict) -> dict:
        """获取预测结果摘要"""
        predicted_labels = results['predicted_labels']
        confidences = [probs[pred] for probs, pred in 
                      zip(results['probabilities'], results['predictions'])]
        
        # 总体统计
        class_counts = pd.Series(predicted_labels).value_counts()
        
        summary = {
            'total_spots': len(predicted_labels),
            'num_classes_predicted': len(class_counts),
            'class_distribution': class_counts.to_dict(),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        # 按样本统计
        df = pd.DataFrame({
            'sample_name': results['sample_names'],
            'predicted_class': predicted_labels,
            'confidence': confidences
        })
        
        sample_stats = []
        for sample in df['sample_name'].unique():
            sample_df = df[df['sample_name'] == sample]
            sample_class_counts = sample_df['predicted_class'].value_counts()
            
            sample_stat = {
                'sample_name': sample,
                'num_spots': len(sample_df),
                'num_classes': len(sample_class_counts),
                'class_distribution': sample_class_counts.to_dict(),
                'average_confidence': sample_df['confidence'].mean(),
                'most_common_class': sample_class_counts.index[0],
                'most_common_class_ratio': sample_class_counts.iloc[0] / len(sample_df)
            }
            sample_stats.append(sample_stat)
        
        summary['sample_statistics'] = sample_stats
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Simple Spot Classifier Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the data')
    parser.add_argument('--samples', type=str, nargs='+', required=True,
                       help='List of sample names to predict')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Do not create visualizations')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建推理器
    inference = SimpleSpotInference(args.model, args.config)
    
    # 进行预测
    results = inference.predict_dataset(
        data_root=args.data_root,
        samples=args.samples,
        output_dir=args.output_dir,
        save_results=not args.no_save,
        visualize=not args.no_visualize
    )
    
    # 打印摘要
    summary = inference.get_prediction_summary(results)
    
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total spots processed: {summary['total_spots']}")
    print(f"Number of classes predicted: {summary['num_classes_predicted']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    print(f"Confidence range: [{summary['min_confidence']:.3f}, {summary['max_confidence']:.3f}]")
    
    print("\nClass Distribution:")
    for class_name, count in summary['class_distribution'].items():
        percentage = (count / summary['total_spots']) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\nSample Statistics:")
    for sample_stat in summary['sample_statistics']:
        print(f"  {sample_stat['sample_name']}:")
        print(f"    Spots: {sample_stat['num_spots']}")
        print(f"    Classes: {sample_stat['num_classes']}")
        print(f"    Most common: {sample_stat['most_common_class']} ({sample_stat['most_common_class_ratio']:.1%})")
        print(f"    Avg confidence: {sample_stat['average_confidence']:.3f}")
    
    if args.output_dir and not args.no_save:
        print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()