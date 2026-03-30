import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.spotformer import SpotResNet, create_spotresnet
from src.data.spotformer_dataset import SpotFormerDataset, get_transforms

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用训练好的SpotFormer模型进行预测')
    parser.add_argument('--config', type=str, default='configs/spotformer_config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--sample', type=str, required=True, help='要预测的样本ID，如A1')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='预测结果保存目录')
    parser.add_argument('--auto_patch_size', action='store_true', help='是否使用自适应patch大小')
    parser.add_argument('--skip_undetermined', action='store_true', help='是否跳过"undetermined"标签的样本')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path, config_path=None, device='cuda'):
    """加载模型"""
    # 加载配置
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 默认配置
        config = {
            'model': {
                'variant': 'base',
                'num_classes': 4,
                'patch_size': 16,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4,
                'dropout': 0.1
            }
        }
    
    # 创建模型
    model_type = config['model'].get('type', 'resnet50')
    if model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model = create_spotresnet(
            num_classes=config['model']['num_classes'],
            model_type=model_type,
            patch_size=config['model']['patch_size'],
            use_spatial_attention=config['model'].get('use_spatial_attention', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config

def predict_sample(model, sample_id, config, output_dir, device, auto_patch_size=False, skip_undetermined=True):
    """对单个样本进行预测"""
    # 创建数据集
    dataset = SpotFormerDataset(
        data_root=config['data']['data_root'],
        samples=[sample_id],
        patch_size=config['model']['patch_size'] if not auto_patch_size else None,
        max_spots_per_sample=config['data']['max_spots_per_sample'],
        transform=get_transforms('test'),
        mode='test',
        auto_patch_size=auto_patch_size,
        skip_undetermined=skip_undetermined
    )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    # 预测
    all_predictions = []
    all_labels = []
    all_positions = []
    all_spot_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            patches = batch['patches'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            
            # 预测
            spot_logits = model(patches, positions, mask)
            spot_probs = F.softmax(spot_logits, dim=-1)
            
            # 获取预测结果
            spot_preds = torch.argmax(spot_probs, dim=-1)
            
            # 只保留有效的spot
            valid_preds = spot_preds[mask]
            valid_labels = batch['labels'][mask].cpu().numpy()
            valid_positions = batch['positions'][mask].cpu().numpy()
            
            # 保存结果
            all_predictions.extend(valid_preds.cpu().numpy())
            all_labels.extend(valid_labels)
            all_positions.extend(valid_positions)
            
            # 获取spot ID
            sample_data = dataset.sample_data[sample_id]
            all_spot_ids.extend(sample_data['spots_data']['spot_id'].tolist())
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果DataFrame
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    results = []
    
    for i, (pred, label, pos, spot_id) in enumerate(zip(all_predictions, all_labels, all_positions, all_spot_ids)):
        results.append({
            'spot_id': spot_id,
            'x': pos[0],
            'y': pos[1],
            'true_label': idx_to_label[label],
            'pred_label': idx_to_label[pred],
            'correct': pred == label
        })
    
    result_df = pd.DataFrame(results)
    
    # 保存预测结果
    output_path = os.path.join(output_dir, f"{sample_id}_predictions.csv")
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    
    # 计算准确率
    accuracy = result_df['correct'].mean()
    print(f"样本 {sample_id} 的预测准确率: {accuracy:.4f}")
    
    # 可视化预测结果
    visualize_predictions(result_df, sample_id, output_dir)
    
    return result_df

def visualize_predictions(result_df, sample_id, output_dir):
    """可视化预测结果"""
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真实标签分布
    true_counts = result_df['true_label'].value_counts()
    ax1.pie(true_counts.values, labels=true_counts.index, autopct='%1.1f%%')
    ax1.set_title(f'{sample_id} - 真实标签分布')
    
    # 预测标签分布
    pred_counts = result_df['pred_label'].value_counts()
    ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
    ax2.set_title(f'{sample_id} - 预测标签分布')
    
    plt.tight_layout()
    
    # 保存图形
    viz_path = os.path.join(output_dir, f"{sample_id}_prediction_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测可视化已保存到: {viz_path}")
    
    # 创建空间分布图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 按标签着色
    labels = result_df['pred_label'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        subset = result_df[result_df['pred_label'] == label]
        ax.scatter(subset['x'], subset['y'], c=[colors[i]], label=label, alpha=0.7, s=50)
    
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'{sample_id} - 预测标签空间分布')
    ax.legend()
    ax.invert_yaxis()  # 反转Y轴，使原点在左上角
    
    # 保存空间分布图
    spatial_path = os.path.join(output_dir, f"{sample_id}_spatial_distribution.png")
    plt.savefig(spatial_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"空间分布图已保存到: {spatial_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载模型
    model, model_config = load_model(
        model_path=args.model_path,
        config_path=args.config,
        device=device
    )
    
    # 预测样本
    results = predict_sample(
        model=model,
        sample_id=args.sample,
        config=config,
        output_dir=args.output_dir,
        device=device,
        auto_patch_size=args.auto_patch_size,
        skip_undetermined=args.skip_undetermined
    )
    
    if results is not None:
        print(f"Prediction completed for sample {args.sample}")
        print(f"Results saved to {args.output_dir}")
    else:
        print(f"Failed to predict sample {args.sample}")

if __name__ == '__main__':
    main()