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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.spotformer import SpotResNet, create_spotresnet
from src.data.spotformer_dataset import create_dataloaders

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题，支持per-class alpha权重"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
        # 处理alpha参数
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        
        # 处理per-class alpha
        if isinstance(self.alpha, torch.Tensor):
            # 如果alpha是tensor且维度大于0，说明是per-class alpha
            if self.alpha.dim() > 0:
                # 确保alpha在正确设备上
                self.alpha = self.alpha.to(inputs.device)
                # 为每个样本获取对应的alpha值
                alpha = self.alpha[targets]
            else:
                alpha = self.alpha
        else:
            alpha = self.alpha
        
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SpotResNetLoss(nn.Module):
    """SpotResNet的损失函数"""
    
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

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class SpotResNetTrainer:
    """SpotResNet训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.setup_logging()
        
        # 创建输出目录
        self.setup_directories()
        
        # 加载数据
        self.setup_data()
        
        # 创建模型
        self.setup_model()
        
        # 设置优化器和调度器
        self.setup_optimizer()
        
        # 设置损失函数
        self.setup_loss()
        
        # 设置早停
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            min_delta=config['training']['early_stopping_min_delta']
        )
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # 混合精度训练
        self.use_mixed_precision = self.config['device'].get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"spotresnet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.config['paths']['output_dir']) / f"spotresnet_{timestamp}"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.visualization_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """设置数据加载器"""
        self.logger.info("Setting up data loaders...")
        
        # 获取配置参数
        auto_patch_size = self.config['data'].get('auto_patch_size', False)
        skip_undetermined = self.config['data'].get('skip_undetermined', False)
        validation_split = self.config['data'].get('validation_split', None)
        
        # 从配置文件中读取欠采样参数
        undersample_majority = self.config['data'].get('undersample_majority', True)
        undersample_ratio = self.config['data'].get('undersample_ratio', 0.5)
        
        # 根据auto_patch_size决定是否传递patch_size参数
        create_dataloaders_kwargs = {
            'data_root': self.config['data']['data_root'],
            'train_samples': self.config['data']['train_samples'],
            'test_samples': self.config['data'].get('test_samples'),
            'batch_size': self.config['data']['batch_size'],
            'max_spots_per_sample': self.config['data']['max_spots_per_sample'],
            'num_workers': self.config['data']['num_workers'],
            'auto_patch_size': auto_patch_size,
            'skip_undetermined': skip_undetermined,
            'validation_split': validation_split,
            'undersample_majority': undersample_majority,  # 从配置文件中读取
            'undersample_ratio': undersample_ratio  # 从配置文件中读取
        }
        
        # 只有在auto_patch_size为False时才传递patch_size
        if not auto_patch_size:
            create_dataloaders_kwargs['patch_size'] = self.config['model']['patch_size']
        
        # 如果配置中仍有val_samples，则传递它
        if 'val_samples' in self.config['data']:
            create_dataloaders_kwargs['val_samples'] = self.config['data']['val_samples']
        
        self.dataloaders, self.label_to_idx, self.idx_to_label = create_dataloaders(**create_dataloaders_kwargs)
        
        self.num_classes = len(self.label_to_idx)
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Label mapping: {self.label_to_idx}")
        if auto_patch_size:
            self.logger.info("Auto patch size enabled")
        else:
            self.logger.info(f"Using fixed patch size: {self.config['model']['patch_size']}")
        if skip_undetermined:
            self.logger.info("Skipping samples with 'undetermined' labels")
        else:
            self.logger.info("Including samples with 'undetermined' labels")
        
        # 记录欠采样配置
        if undersample_majority:
            self.logger.info(f"Undersampling enabled for majority classes with ratio: {undersample_ratio}")
        else:
            self.logger.info("Undersampling disabled")
        
        # 计算并记录训练集的类别分布
        self.calculate_class_distribution()
    
    def calculate_class_distribution(self):
        """计算训练集的类别分布"""
        self.logger.info("Calculating class distribution...")
        
        # 初始化类别计数器
        class_counts = {label: 0 for label in self.label_to_idx.keys()}
        
        # 遍历训练数据集统计类别数量
        train_dataset = self.dataloaders['train'].dataset
        for sample_data in train_dataset.sample_data:
            spots_data = sample_data['spots_data']
            label_counts = spots_data['label'].value_counts()
            
            for label, count in label_counts.items():
                if label in class_counts:
                    class_counts[label] += count
        
        # 记录类别分布
        self.logger.info(f"Training set class distribution: {class_counts}")
        
        # 保存类别分布信息，用于后续自动计算权重
        self.class_counts = class_counts
        self.total_samples = sum(class_counts.values())
        
        # 如果配置中启用了自动计算类别权重，则计算权重
        if self.config['loss'].get('auto_class_weights', False):
            self.auto_calculate_class_weights()
    
    def auto_calculate_class_weights(self):
        """根据类别分布自动计算类别权重"""
        self.logger.info("Auto-calculating class weights based on class distribution...")
        
        # 计算每个类别的权重
        # 使用逆频率加权，即样本数越少的类别权重越高
        class_weights = []
        for label in sorted(self.label_to_idx.keys()):
            count = self.class_counts.get(label, 0)
            if count > 0:
                # 使用逆频率作为权重，并添加平滑因子避免除零
                weight = self.total_samples / (self.num_classes * count)
                # 可以根据需要调整权重计算方式，例如使用平方根或对数
                # weight = np.sqrt(self.total_samples / (self.num_classes * count))
                # weight = np.log1p(self.total_samples / count)
            else:
                weight = 1.0  # 如果类别没有样本，使用默认权重
            
            class_weights.append(weight)
        
        # 归一化权重，使平均权重为1.0
        class_weights = np.array(class_weights)
        class_weights = class_weights / np.mean(class_weights)
        
        # 将计算出的权重更新到配置中
        self.config['loss']['class_weights'] = class_weights.tolist()
        
        self.logger.info(f"Auto-calculated class weights: {class_weights.tolist()}")
        self.logger.info(f"Class counts: {[self.class_counts.get(label, 0) for label in sorted(self.label_to_idx.keys())]}")
        
        # 返回计算出的权重
        return class_weights
    
    def setup_model(self):
        """设置模型"""
        self.logger.info("Setting up model...")
        
        model_type = self.config['model']['type']
        
        # 支持轻量化ViT模型和ResNet模型
        self.logger.info(f"Creating model: {model_type}")
        self.model = create_spotresnet(
            num_classes=self.num_classes,
            model_type=model_type,
            patch_size=self.config['model']['patch_size']
        )
        
        self.model = self.model.to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model_type}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_optimizer(self):
        """
        设置优化器和学习率调度器
        """
        # 获取优化器配置
        optimizer_config = self.config['optimizer']
        lr = optimizer_config['learning_rate']
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        # 优化所有参数
        self.logger.info("Setting up optimizer for all parameters")
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 创建优化器
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_config['type'] == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
        
        # 设置学习率调度器
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience']
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {optimizer_config['type']}, LR: {lr}")
        if self.scheduler:
            self.logger.info(f"Scheduler: {scheduler_config['type']}")
    
    def setup_loss(self):
        """设置损失函数"""
        loss_config = self.config['loss']
        
        # 获取类别权重
        class_weights = loss_config.get('class_weights', None)
        if class_weights is not None:
            # 确保类别权重与类别数量匹配
            if len(class_weights) != self.num_classes:
                self.logger.warning(f"类别权重数量({len(class_weights)})与类别数量({self.num_classes})不匹配，将不使用类别权重")
                class_weights = None
            else:
                # 转换为tensor并移动到设备
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        self.criterion = SpotResNetLoss(
            use_focal=loss_config.get('type') == 'focal',
            focal_alpha=loss_config.get('alpha', 0.25),  # 使用配置文件中的alpha值
            focal_gamma=loss_config.get('gamma', 3.0),   # 使用配置文件中的gamma值
            class_weights=class_weights  # 添加类别权重参数
        )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            patches = batch['patches'].to(self.device)
            labels = batch['labels'].to(self.device)
            positions = batch['positions'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # 混合精度训练
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # 前向传播
                    spot_logits = self.model(patches, positions, mask)
                    
                    # 计算损失
                    loss = self.criterion(spot_logits, labels, mask)
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clip_norm'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_norm']
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 前向传播
                spot_logits = self.model(patches, positions, mask)
                
                # 计算损失
                loss = self.criterion(spot_logits, labels, mask)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.config['training']['gradient_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_norm']
                    )
                
                self.optimizer.step()
            
            # 统计
            batch_size = patches.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}"
            })
            
            # 记录到TensorBoard
            global_step = self.current_epoch * len(self.dataloaders['train']) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / total_samples
        
        return avg_loss
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_spot_predictions = []
        all_spot_labels = []
        all_spot_probs = []  # 存储预测概率
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloaders['val'], desc="Validation")):
                # 移动数据到设备
                patches = batch['patches'].to(self.device)
                labels = batch['labels'].to(self.device)
                positions = batch['positions'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # 混合精度训练
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        spot_logits = self.model(patches, positions, mask)
                        
                        # 计算损失
                        loss = self.criterion(spot_logits, labels, mask)
                else:
                    # 前向传播
                    spot_logits = self.model(patches, positions, mask)
                
                # 计算损失
                loss = self.criterion(spot_logits, labels, mask)
                
                # 统计
                batch_size = patches.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # 收集spot级别预测结果
                spot_probs = torch.softmax(spot_logits, dim=-1)  # 获取预测概率
                spot_predictions = torch.argmax(spot_logits, dim=-1)
                
                for i in range(batch_size):
                    valid_mask = mask[i]
                    valid_preds = spot_predictions[i][valid_mask].cpu().numpy()
                    valid_labels = labels[i][valid_mask].cpu().numpy()
                    valid_probs = spot_probs[i][valid_mask].cpu().numpy()
                    
                    all_spot_predictions.extend(valid_preds)
                    all_spot_labels.extend(valid_labels)
                    all_spot_probs.extend(valid_probs)
        
        avg_loss = total_loss / total_samples
        
        # 计算spot级别精度
        spot_accuracy = accuracy_score(all_spot_labels, all_spot_predictions)
        
        # 输出详细的预测统计信息
        self.logger.info(f"\n--- Validation Prediction Statistics ---")
        self.logger.info(f"Total validation spots: {len(all_spot_labels)}")
        self.logger.info(f"Unique true labels: {np.unique(all_spot_labels)}")
        self.logger.info(f"Unique predicted labels: {np.unique(all_spot_predictions)}")
        
        # 输出每个类别的预测统计和示例
        for label_idx in range(len(self.idx_to_label)):
            label_name = self.idx_to_label[label_idx]
            true_count = sum(1 for l in all_spot_labels if l == label_idx)
            pred_count = sum(1 for p in all_spot_predictions if p == label_idx)
            correct_count = sum(1 for t, p in zip(all_spot_labels, all_spot_predictions) if t == p == label_idx)
            
            if true_count > 0:
                precision = correct_count / pred_count if pred_count > 0 else 0
                recall = correct_count / true_count
                self.logger.info(f"Class {label_name}: True={true_count}, Pred={pred_count}, Correct={correct_count}, Precision={precision:.3f}, Recall={recall:.3f}")
                
                # 为每个类别输出预测示例
                self.logger.info(f"  Examples for class {label_name}:")
                example_count = 0
                
                # 找出正确预测的示例
                for i, (true_label, pred_label, probs) in enumerate(zip(all_spot_labels, all_spot_predictions, all_spot_probs)):
                    if true_label == pred_label == label_idx and example_count < 2:
                        confidence = probs[pred_label]
                        self.logger.info(f"    Correct: Pred={label_name}({confidence:.3f}), True={label_name}")
                        example_count += 1
                
                # 找出错误预测的示例（真实为该类别但预测错误）
                example_count = 0
                for i, (true_label, pred_label, probs) in enumerate(zip(all_spot_labels, all_spot_predictions, all_spot_probs)):
                    if true_label == label_idx and pred_label != label_idx and example_count < 2:
                        pred_label_name = self.idx_to_label[pred_label]
                        confidence = probs[pred_label]
                        true_confidence = probs[label_idx]
                        self.logger.info(f"    Incorrect: Pred={pred_label_name}({confidence:.3f}), True={label_name}(True confidence: {true_confidence:.3f})")
                        example_count += 1
        
        # 返回spot级别精度作为主要指标
        return avg_loss, spot_accuracy, all_spot_predictions, all_spot_labels
    
    def save_main_classifier_predictions(self):
        """
        保存主分类模块对所有数据的预测结果，用于第二阶段训练空间矫正模块
        """
        self.logger.info("Generating predictions from main classifier...")
        
        # 设置模型为评估模式
        self.model.eval()
        self.model.use_correction = False  # 确保只使用主分类模块
        
        all_predictions = {}
        all_positions = {}
        
        # 处理训练集
        with torch.no_grad():
            for batch_idx, (patches, positions, labels, mask) in enumerate(self.train_loader):
                patches = patches.to(self.device)
                positions = positions.to(self.device)
                
                # 使用主分类模块进行预测
                spot_logits = self.model(patches, positions, mask)
                spot_probs = torch.softmax(spot_logits, dim=-1)
                spot_preds = torch.argmax(spot_probs, dim=-1)
                
                # 保存预测结果和位置信息
                for i in range(patches.size(0)):
                    sample_id = f"train_{batch_idx}_{i}"
                    all_predictions[sample_id] = spot_preds[i].cpu().numpy()
                    all_positions[sample_id] = positions[i].cpu().numpy()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed {batch_idx}/{len(self.train_loader)} training batches")
        
        # 处理验证集
        with torch.no_grad():
            for batch_idx, (patches, positions, labels, mask) in enumerate(self.val_loader):
                patches = patches.to(self.device)
                positions = positions.to(self.device)
                
                # 使用主分类模块进行预测
                spot_logits = self.model(patches, positions, mask)
                spot_probs = torch.softmax(spot_logits, dim=-1)
                spot_preds = torch.argmax(spot_probs, dim=-1)
                
                # 保存预测结果和位置信息
                for i in range(patches.size(0)):
                    sample_id = f"val_{batch_idx}_{i}"
                    all_predictions[sample_id] = spot_preds[i].cpu().numpy()
                    all_positions[sample_id] = positions[i].cpu().numpy()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed {batch_idx}/{len(self.val_loader)} validation batches")
        
        # 保存预测结果和位置信息
        predictions_path = os.path.join(self.checkpoint_dir, 'main_classifier_predictions.pth')
        torch.save({
            'predictions': all_predictions,
            'positions': all_positions,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }, predictions_path)
        
        self.logger.info(f"Saved main classifier predictions to {predictions_path}")
        
        return predictions_path

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            self.logger.info(f"Saved best checkpoint at epoch {epoch + 1}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")

        # 记录训练开始时间
        start_time = time.time()
        
        # 标准训练流程
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_accuracy, val_predictions, val_labels = self.validate_epoch()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_accuracy)
                else:
                    self.scheduler.step()
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # TensorBoard记录
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            is_best = val_accuracy > self.best_val_acc
            if is_best:
                self.best_val_acc = val_accuracy
            
            self.save_checkpoint(epoch, is_best)
            
            # 日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}, Best Val Acc: {self.best_val_acc:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # 每10个epoch输出一次详细信息
            if (epoch + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(f"Training time elapsed: {elapsed_time/3600:.2f} hours")
            
            # 早停检查
            if self.early_stopping(val_accuracy, self.model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 训练结束后的评估和可视化
        self.final_evaluation()
        self.plot_training_curves()
        
        # 记录总训练时间
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours!")
    
    def final_evaluation(self):
        """最终评估"""
        self.logger.info("Performing final evaluation...")
        
        # 加载最佳模型
        best_checkpoint = torch.load(self.checkpoint_dir / 'best.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # 验证集评估
        val_loss, val_accuracy, val_predictions, val_labels = self.validate_epoch()
        
        # 获取验证集中实际存在的类别
        unique_labels = sorted(np.unique(val_labels + val_predictions))
        class_names = [self.idx_to_label[i] for i in unique_labels]
        
        # 分类报告
        report = classification_report(val_labels, val_predictions, labels=unique_labels, target_names=class_names)
        
        self.logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        self.logger.info(f"Classification report:\n{report}")
        
        # 保存报告
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(f"Final validation accuracy: {val_accuracy:.4f}\n\n")
            f.write(f"Classification report:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(val_labels, val_predictions, labels=unique_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'training_curves.png', dpi=300)
        plt.close()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train SpotResNet model')
    parser.add_argument('--config', type=str, default='./configs/spotformer_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = SpotResNetTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed training from epoch {trainer.current_epoch}")
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()