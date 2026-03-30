import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import random

class SpotFormerDataset(Dataset):
    """SpotFormer专用数据集，直接处理spot级别的分类任务"""
    
    def __init__(self, 
             data_root: str,
             samples: List[str],
             patch_size: int = 64,
             max_spots_per_sample: int = 1000,
             mode: str = 'train',
             auto_patch_size: bool = False,
             skip_undetermined: bool = True,
             undersample_majority: bool = False,
             undersample_ratio: float = 0.5,
             val_spots_file: Optional[str] = None):
        """
        Args:
            data_root: 数据根目录
            samples: 样本列表，如 ['A1', 'B1', 'C1']
            patch_size: 每个spot的patch大小（仅在auto_patch_size=False时使用）
            max_spots_per_sample: 每个样本最大spot数量
            mode: 'train', 'val', 'test'
            auto_patch_size: 是否启用自适应patch size
            skip_undetermined: 是否跳过"undetermined"标签的样本
            undersample_majority: 是否对多数类进行欠采样
            undersample_ratio: 欠采样比例，多数类保留的比例
        """
        self.data_root = data_root
        self.samples = samples
        self.patch_size = patch_size
        self.max_spots_per_sample = max_spots_per_sample
        self.mode = mode
        self.auto_patch_size = auto_patch_size
        self.skip_undetermined = skip_undetermined
        self.undersample_majority = undersample_majority
        self.undersample_ratio = undersample_ratio
        self.val_spots_file = val_spots_file
        
        # 路径设置
        self.img_dir = os.path.join(data_root, 'ST-imgs')
        self.label_dir = os.path.join(data_root, 'ST-pat', 'lbl')
        
        # 加载验证集spot索引（如果有）
        self.val_spot_indices = None
        self.all_spots = None
        if self.val_spots_file and os.path.exists(self.val_spots_file):
            import pickle
            try:
                with open(self.val_spots_file, 'rb') as f:
                    data = pickle.load(f)
                    self.val_spot_indices = data.get('val_spot_indices', [])
                    self.all_spots = data.get('all_spots', [])
                logging.info(f"已加载验证集spot索引，共{len(self.val_spot_indices) if self.val_spot_indices else 0}个spot")
            except Exception as e:
                logging.error(f"加载验证集spot索引失败: {str(e)}")
        
        # 加载所有数据
        self.sample_data = self._load_all_data()
        
        # 创建标签映射
        self.label_to_idx, self.idx_to_label = self._create_label_mapping()
        self.num_classes = len(self.label_to_idx)
        
        # 如果启用自适应patch size，计算每个样本的相邻距离
        if self.auto_patch_size:
            self.sample_patch_sizes = self._calculate_sample_patch_sizes()
        else:
            # 使用固定的patch size
            self.sample_patch_sizes = {sample_id: self.patch_size for sample_id in self.samples}
        
        logging.info(f"Loaded {len(self.sample_data)} samples with {self.num_classes} classes")
        logging.info(f"Label mapping: {self.label_to_idx}")
        if self.auto_patch_size:
            logging.info(f"Sample patch sizes: {self.sample_patch_sizes}")
    
    def _load_all_data(self) -> List[Dict]:
        """加载所有样本的数据"""
        all_data = []
        
        for sample_id in self.samples:
            try:
                # 加载图像
                img_path = self._get_image_path(sample_id)
                if not os.path.exists(img_path):
                    logging.warning(f"Image not found: {img_path}")
                    continue
                
                # 加载标签
                label_path = os.path.join(self.label_dir, f"{sample_id}_labeled_coordinates.tsv")
                if not os.path.exists(label_path):
                    logging.warning(f"Label file not found: {label_path}")
                    continue
                
                # 读取标签数据
                label_df = pd.read_csv(label_path, sep='\t')
                
                # 过滤有效数据
                valid_spots = label_df.dropna(subset=['pixel_x', 'pixel_y', 'label'])
                
                if len(valid_spots) == 0:
                    logging.warning(f"No valid spots found in {sample_id}")
                    continue
                
                # 如果过滤后没有有效spot，跳过该样本
                if len(valid_spots) == 0:
                    logging.warning(f"No valid spots after filtering in {sample_id}")
                    continue
                
                # 根据模式和验证集spot索引过滤数据
                if self.val_spot_indices is not None and self.all_spots is not None:
                    if self.mode == 'val':
                        # 验证集：只保留在val_spot_indices中的spot
                        # 创建一个集合用于快速查找
                        val_spot_set = set()
                        for idx in self.val_spot_indices:
                            spot_sample_id, x, y, label = self.all_spots[idx]
                            if spot_sample_id == sample_id:
                                # 将坐标和标签组合作为唯一标识
                                val_spot_set.add((x, y, label))
                        
                        # 过滤验证集spot
                        original_count = len(valid_spots)
                        filtered_spots = []
                        for _, spot in valid_spots.iterrows():
                            if (spot['pixel_x'], spot['pixel_y'], spot['label']) in val_spot_set:
                                filtered_spots.append(spot)
                        
                        valid_spots = pd.DataFrame(filtered_spots)
                        logging.info(f"验证集 {sample_id}: 保留了 {len(valid_spots)} 个spot (原始 {original_count} 个)")
                    else:
                        # 训练集：排除在val_spot_indices中的spot
                        # 创建一个集合用于快速查找
                        val_spot_set = set()
                        for idx in self.val_spot_indices:
                            spot_sample_id, x, y, label = self.all_spots[idx]
                            if spot_sample_id == sample_id:
                                # 将坐标和标签组合作为唯一标识
                                val_spot_set.add((x, y, label))
                        
                        # 过滤掉验证集spot
                        original_count = len(valid_spots)
                        filtered_spots = []
                        for _, spot in valid_spots.iterrows():
                            if (spot['pixel_x'], spot['pixel_y'], spot['label']) not in val_spot_set:
                                filtered_spots.append(spot)
                        
                        valid_spots = pd.DataFrame(filtered_spots)
                        logging.info(f"训练集 {sample_id}: 保留了 {len(valid_spots)} 个spot (排除了 {original_count - len(valid_spots)} 个验证集spot)")
                
                # 如果过滤后没有有效spot，跳过该样本
                if len(valid_spots) == 0:
                    logging.warning(f"No valid spots after val/train split in {sample_id}")
                    continue
                
                # 如果启用欠采样且是训练模式
                if self.undersample_majority and self.mode == 'train':
                    # 统计各类别数量
                    label_counts = valid_spots['label'].value_counts()
                    if len(label_counts) > 1:  # 确保有多个类别
                        # 找出多数类
                        majority_class = label_counts.index[0]
                        majority_count = label_counts.iloc[0]
                        
                        # 计算欠采样后的数量
                        target_count = int(majority_count * self.undersample_ratio)
                        
                        # 对多数类进行欠采样
                        majority_spots = valid_spots[valid_spots['label'] == majority_class]
                        minority_spots = valid_spots[valid_spots['label'] != majority_class]
                        
                        if len(majority_spots) > target_count:
                            # 随机选择target_count个多数类样本
                            majority_spots = majority_spots.sample(n=target_count, random_state=42)
                            logging.info(f"Sample {sample_id}: Undersampled majority class '{majority_class}' from {majority_count} to {target_count} spots")
                            logging.info(f"Sample {sample_id}: Original class distribution: {dict(label_counts)}")
                            logging.info(f"Sample {sample_id}: New class distribution: {dict(pd.concat([majority_spots, minority_spots])['label'].value_counts())}")
                        
                        # 合并欠采样后的数据
                        valid_spots = pd.concat([majority_spots, minority_spots])
                
                # 限制spot数量
                if len(valid_spots) > self.max_spots_per_sample:
                    valid_spots = valid_spots.sample(n=self.max_spots_per_sample, random_state=42)
                
                sample_data = {
                    'sample_id': sample_id,
                    'image_path': img_path,
                    'spots_data': valid_spots.reset_index(drop=True)
                }
                
                all_data.append(sample_data)
                logging.info(f"Loaded {sample_id}: {len(valid_spots)} spots")
                
            except Exception as e:
                logging.error(f"Error loading {sample_id}: {str(e)}")
                continue
        
        return all_data
    
    def _calculate_sample_patch_sizes(self) -> Dict[str, int]:
        """计算每个样本的自适应patch size，基于相邻spot的平均距离"""
        sample_patch_sizes = {}
        
        for sample_data in self.sample_data:
            sample_id = sample_data['sample_id']
            spots = sample_data['spots_data']
            
            # 计算相邻spot的距离（在网格中相邻的spot）
            distances = []
            
            # 创建一个网格坐标到索引的映射，便于查找相邻spot
            coord_to_idx = {}
            for i in range(len(spots)):
                # 使用x,y坐标（网格坐标）进行四舍五入作为索引
                x, y = round(spots.iloc[i]['x']), round(spots.iloc[i]['y'])
                coord_to_idx[(x, y)] = i
            
            # 对每个spot查找其相邻spot
            for i in range(len(spots)):
                # 使用x,y坐标（网格坐标）进行四舍五入判断相邻关系
                x, y = round(spots.iloc[i]['x']), round(spots.iloc[i]['y'])
                
                # 查找四个方向的相邻spot：上、下、左、右
                adjacent_coords = [
                    (x, y-1),  # 上
                    (x, y+1),  # 下
                    (x-1, y),  # 左
                    (x+1, y)   # 右
                ]
                
                for adj_x, adj_y in adjacent_coords:
                    if (adj_x, adj_y) in coord_to_idx:
                        j = coord_to_idx[(adj_x, adj_y)]
                        # 计算实际像素距离（使用pixel_x,pixel_y）
                        x1, y1 = spots.iloc[i]['pixel_x'], spots.iloc[i]['pixel_y']
                        x2, y2 = spots.iloc[j]['pixel_x'], spots.iloc[j]['pixel_y']
                        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        distances.append(distance)
            
            # 计算平均距离
            if distances:
                avg_distance = np.mean(distances)
                # 根据平均距离计算patch size，确保至少为32，最大为256
                # 使用更合理的计算方式，避免所有样本都变成最大值
                patch_size = max(32, min(256, int(avg_distance * 1.5)))  # 降低倍数
                logging.info(f"Sample {sample_id}: avg_distance={avg_distance:.2f}, calculated patch_size={patch_size}")
            else:
                # 如果没有足够的相邻spot计算距离，使用默认值
                patch_size = self.patch_size
                logging.info(f"Sample {sample_id}: no adjacent distances calculated, using default patch_size={patch_size}")
            
            sample_patch_sizes[sample_id] = patch_size
        
        return sample_patch_sizes
      
    def _get_image_path(self, sample_id: str) -> str:
        letter = sample_id[0]  # A, B, C, etc.
        img_dir_path = os.path.join(self.img_dir, letter, sample_id)
        
        # 查找图像文件
        if os.path.exists(img_dir_path):
            for file in os.listdir(img_dir_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                    return os.path.join(img_dir_path, file)
        
        raise FileNotFoundError(f"No image found for sample {sample_id}")
    
    def _create_label_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """创建标签到索引的映射"""
        all_labels = set()
        
        for sample_data in self.sample_data:
            labels = sample_data['spots_data']['label'].unique()
            all_labels.update(labels)
        
        sorted_labels = sorted(list(all_labels))
        
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        return label_to_idx, idx_to_label
    
    def _extract_patches(self, image: np.ndarray, spots_data: pd.DataFrame, sample_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从图像中提取spot对应的patches"""
        patches = []
        labels = []
        positions = []
        
        # 获取当前样本的patch size
        if self.auto_patch_size and sample_id in self.sample_patch_sizes:
            current_patch_size = self.sample_patch_sizes[sample_id]
        else:
            current_patch_size = self.patch_size
            
        for _, spot in spots_data.iterrows():
            x_center = int(spot['pixel_x'])
            y_center = int(spot['pixel_y'])
            
            # 提取patch
            patch = self._extract_patch(image, x_center, y_center, current_patch_size)
            
            patches.append(patch)
            labels.append(self.label_to_idx[spot['label']])
            positions.append([x_center, y_center])
        
        return np.array(patches), np.array(labels), np.array(positions)
    
    def _extract_patch(self, image: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
        """提取单个patch，处理边界情况
        
        Args:
            image: 输入图像
            x, y: spot中心坐标
            patch_size: patch大小
            
        Returns:
            patch: 提取的patch, shape=(patch_size, patch_size, 3)
        """
        h, w = image.shape[:2]
        half_size = patch_size // 2
        
        # 计算patch的边界，确保spot位于中心
        x_center = int(x)
        y_center = int(y)
        
        # 计算patch的边界
        x_min = x_center - half_size
        x_max = x_center + half_size
        y_min = y_center - half_size
        y_max = y_center + half_size
        
        # 处理边界情况，确保patch不超出图像范围
        if x_min < 0:
            x_max += -x_min  # 向右移动
            x_min = 0
        elif x_max > w:
            x_min -= x_max - w  # 向左移动
            x_max = w
            
        if y_min < 0:
            y_max += -y_min  # 向下移动
            y_min = 0
        elif y_max > h:
            y_min -= y_max - h  # 向上移动
            y_max = h
        
        # 确保边界值在有效范围内
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        # 提取patch
        patch = image[y_min:y_max, x_min:x_max]
        
        # 如果patch小于目标大小，进行填充
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            # 计算需要填充的大小
            pad_h = max(0, patch_size - patch.shape[0])
            pad_w = max(0, patch_size - patch.shape[1])
            
            # 对称填充
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if len(patch.shape) == 3:
                patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                             mode='reflect')
            else:
                patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                             mode='reflect')
            
        return patch
    
    def _apply_transforms(self, patches: np.ndarray) -> torch.Tensor:
        """应用基本变换（归一化和转换为tensor）"""
        # 转换为浮点数并归一化到[0,1]范围
        patches_float = patches.astype(np.float32) / 255.0
        
        # 应用ImageNet标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized_patches = []
        for patch in patches_float:
            # 转换通道顺序 HxWxC -> CxHxW
            patch = np.transpose(patch, (2, 0, 1))
            # 应用标准化
            normalized_patch = (patch - mean[:, None, None]) / std[:, None, None]
            normalized_patches.append(normalized_patch)
        
        return torch.tensor(np.stack(normalized_patches), dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.sample_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_data = self.sample_data[idx]
        sample_id = sample_data['sample_id']
        
        # 加载图像
        image = cv2.imread(sample_data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        spots_data = sample_data['spots_data']
        
        # 提取patches
        patches, labels, positions = self._extract_patches(image, spots_data, sample_id)
        
        # 应用变换
        patches_tensor = self._apply_transforms(patches)
        
        # 准备标签
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 准备位置信息
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        
        # 准备空间坐标（归一化）
        spatial_coords = torch.tensor(
            spots_data[['x', 'y']].values, 
            dtype=torch.float32
        )
        
        # 创建mask（所有spot都是有效的）
        num_spots = len(spots_data)
        mask = torch.ones(num_spots, dtype=torch.bool)
        
        return {
            'patches': patches_tensor,  # [num_spots, 3, patch_size, patch_size]
            'labels': labels_tensor,    # [num_spots]
            'positions': positions_tensor,     # [num_spots, 2] - pixel coordinates
            'spatial_coords': spatial_coords,  # [num_spots, 2] - spatial coordinates
            'mask': mask,              # [num_spots]
            'sample_id': sample_id,
            'num_spots': num_spots
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """自定义collate函数，处理不同数量的spots"""
        # 找到最大spot数量
        max_spots = max(item['num_spots'] for item in batch)
        batch_size = len(batch)
        patch_size = batch[0]['patches'].shape[-1]
        
        # 初始化批次张量
        batch_patches = torch.zeros(batch_size, max_spots, 3, patch_size, patch_size)
        batch_labels = torch.zeros(batch_size, max_spots, dtype=torch.long)
        batch_positions = torch.zeros(batch_size, max_spots, 2)
        batch_spatial_coords = torch.zeros(batch_size, max_spots, 2)
        batch_mask = torch.zeros(batch_size, max_spots, dtype=torch.bool)
        
        sample_ids = []
        num_spots_list = []
        
        for i, item in enumerate(batch):
            num_spots = item['num_spots']
            
            # 填充数据
            batch_patches[i, :num_spots] = item['patches']
            batch_labels[i, :num_spots] = item['labels']
            batch_positions[i, :num_spots] = item['positions']
            batch_spatial_coords[i, :num_spots] = item['spatial_coords']
            batch_mask[i, :num_spots] = item['mask']
            
            sample_ids.append(item['sample_id'])
            num_spots_list.append(num_spots)
        
        return {
            'patches': batch_patches,
            'labels': batch_labels,
            'positions': batch_positions,
            'spatial_coords': batch_spatial_coords,
            'mask': batch_mask,
            'sample_ids': sample_ids,
            'num_spots': torch.tensor(num_spots_list)
        }
    
    def visualize_sample_spots(self, sample_idx: int = 0, show_all_spots: bool = True, show_patches: bool = False, 
                              spot_marker_size: int = 15, use_star_marker: bool = True):
        """
        可视化样本中的spot位置，显示所有spot并使用清晰的图标标注
        
        Args:
            sample_idx: 样本索引
            show_all_spots: 是否显示所有spot（默认为True）
            show_patches: 是否显示patch边界框
            spot_marker_size: spot标记的大小
            use_star_marker: 是否使用星形标记（否则使用圆形）
        """
        if sample_idx >= len(self.sample_data):
            raise ValueError(f"Sample index {sample_idx} out of range. Dataset has {len(self.sample_data)} samples.")
        
        sample_data = self.sample_data[sample_idx]
        sample_id = sample_data['sample_id']
        spots_data = sample_data['spots_data']
        
        # 加载图像
        image = cv2.imread(sample_data['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取当前样本的patch size
        if self.auto_patch_size and sample_id in self.sample_patch_sizes:
            current_patch_size = self.sample_patch_sizes[sample_id]
        else:
            current_patch_size = self.patch_size
            
        print(f"Visualizing sample {sample_id} with {len(spots_data)} spots")
        print(f"Image shape: {image.shape}")
        print(f"Patch size: {current_patch_size}")
        print(f"Auto patch size: {self.auto_patch_size}")
        
        # 显示所有spot（不再限制数量）
        spots_to_show = spots_data
        print(f"Showing all {len(spots_data)} spots")
        
        # 创建可视化图像（用于显示）
        vis_image = image.copy()
        
        # 为不同标签分配颜色
        unique_labels = spots_to_show['label'].unique()
        label_colors = {}
        # 使用更鲜明的颜色方案
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_labels))))
        for i, label in enumerate(unique_labels):
            # 如果标签超过20个，循环使用颜色
            color_idx = i % 20
            label_colors[label] = tuple(int(c*255) for c in colors[color_idx][:3])
        
        # 在图像上绘制spot位置
        for _, spot in spots_to_show.iterrows():
            x_center = int(spot['pixel_x'])
            y_center = int(spot['pixel_y'])
            label = spot['label']
            
            # 使用更清晰的标记
            color = label_colors[label]
            
            if use_star_marker:
                # 使用星形标记
                cv2.drawMarker(vis_image, (x_center, y_center), color, 
                              markerType=cv2.MARKER_STAR, markerSize=spot_marker_size, thickness=2)
            else:
                # 使用圆形标记
                cv2.circle(vis_image, (x_center, y_center), spot_marker_size, color, -1)
                cv2.circle(vis_image, (x_center, y_center), spot_marker_size, (255, 255, 255), 2)  # 白色边框
            
            # 绘制patch边界框（如果启用）
            if show_patches:
                half_size = current_patch_size // 2
                top_left = (x_center - half_size, y_center - half_size)
                bottom_right = (x_center + half_size, y_center + half_size)
                cv2.rectangle(vis_image, top_left, bottom_right, color, 2)
        
        # 创建更清晰的可视化布局
        plt.figure(figsize=(24, 12))
        
        # 显示完整图像和spot位置
        plt.subplot(1, 3, 1)
        plt.imshow(vis_image)
        plt.title(f'Sample {sample_id} - All Spots ({len(spots_to_show)} total)', fontsize=16, weight='bold')
        plt.axis('off')
        
        # 显示图像的局部放大区域（选择spots密集的区域）
        plt.subplot(1, 3, 2)
        if len(spots_to_show) > 0:
            # 计算spots密度最高的区域
            x_coords = spots_to_show['pixel_x'].astype(int)
            y_coords = spots_to_show['pixel_y'].astype(int)
            
            # 创建一个简化的密度图来找到最密集的区域
            if len(spots_to_show) > 100:
                # 如果spots很多，使用采样来提高性能
                sample_spots = spots_to_show.sample(n=100, random_state=42)
                x_coords = sample_spots['pixel_x'].astype(int)
                y_coords = sample_spots['pixel_y'].astype(int)
            
            # 找到中心区域
            center_x = int(x_coords.median())
            center_y = int(y_coords.median())
            
            # 定义放大区域（600x600）
            crop_size = 600
            half_crop = crop_size // 2
            
            x_min = max(0, center_x - half_crop)
            x_max = min(image.shape[1], center_x + half_crop)
            y_min = max(0, center_y - half_crop)
            y_max = min(image.shape[0], center_y + half_crop)
            
            # 确保裁剪区域大小正确
            if x_max - x_min < crop_size:
                if x_min == 0:
                    x_max = min(image.shape[1], x_min + crop_size)
                else:
                    x_min = max(0, x_max - crop_size)
            
            if y_max - y_min < crop_size:
                if y_min == 0:
                    y_max = min(image.shape[0], y_min + crop_size)
                else:
                    y_min = max(0, y_max - crop_size)
            
            cropped_image = image[y_min:y_max, x_min:x_max].copy()
            
            # 在裁剪图像上绘制spots
            for _, spot in spots_to_show.iterrows():
                x_center = int(spot['pixel_x'])
                y_center = int(spot['pixel_y'])
                
                # 检查spot是否在裁剪区域内
                if x_min <= x_center <= x_max and y_min <= y_center <= y_max:
                    # 调整坐标到裁剪图像
                    x_adj = x_center - x_min
                    y_adj = y_center - y_min
                    label = spot['label']
                    color = label_colors[label]
                    
                    # 使用更大的标记
                    if use_star_marker:
                        cv2.drawMarker(cropped_image, (x_adj, y_adj), color, 
                                      markerType=cv2.MARKER_STAR, markerSize=spot_marker_size+5, thickness=3)
                    else:
                        cv2.circle(cropped_image, (x_adj, y_adj), spot_marker_size+5, color, -1)
                        cv2.circle(cropped_image, (x_adj, y_adj), spot_marker_size+5, (255, 255, 255), 3)
                    
                    # 绘制patch边界框
                    if show_patches:
                        half_size = current_patch_size // 2
                        top_left = (x_adj - half_size, y_adj - half_size)
                        bottom_right = (x_adj + half_size, y_adj + half_size)
                        cv2.rectangle(cropped_image, top_left, bottom_right, color, 2)
            
            plt.imshow(cropped_image)
            plt.title(f'Detailed View (Center: {center_x}, {center_y})', fontsize=16, weight='bold')
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No spots to show', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Detailed View', fontsize=16, weight='bold')
            plt.axis('off')
        
        # 显示图例和统计信息
        plt.subplot(1, 3, 3)
        plt.axis('off')
        
        # 创建图例
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255) 
                          for color in label_colors.values()]
        
        # 显示图例（分两列显示，避免重叠）
        num_labels = len(label_colors)
        if num_labels > 10:
            # 分两列显示
            legend1 = plt.legend(legend_elements[:num_labels//2], list(label_colors.keys())[:num_labels//2], 
                               loc='upper center', bbox_to_anchor=(0.5, 0.95),
                               title='Spot Labels (1/2)', fontsize=9, title_fontsize=11)
            plt.gca().add_artist(legend1)
            legend2 = plt.legend(legend_elements[num_labels//2:], list(label_colors.keys())[num_labels//2:], 
                               loc='upper center', bbox_to_anchor=(0.5, 0.65),
                               title='Spot Labels (2/2)', fontsize=9, title_fontsize=11)
        else:
            # 单列显示
            legend = plt.legend(legend_elements, label_colors.keys(), loc='upper center', 
                               bbox_to_anchor=(0.5, 0.95),
                               title='Spot Labels', fontsize=10, title_fontsize=12)
        
        # 添加统计信息
        plt.text(0.5, 0.5, f'Total Spots: {len(spots_data)}', 
                transform=plt.gca().transAxes, ha='center', fontsize=14, weight='bold')
        
        plt.text(0.5, 0.45, f'Image Size: {image.shape[1]}×{image.shape[0]}', 
                transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        plt.text(0.5, 0.4, f'Patch Size: {current_patch_size}', 
                transform=plt.gca().transAxes, ha='center', fontsize=11)
        
        # 显示标签分布
        label_counts = spots_to_show['label'].value_counts()
        plt.text(0.5, 0.35, 'Top Labels:', transform=plt.gca().transAxes, 
                ha='center', fontsize=13, weight='bold')
        
        y_pos = 0.3
        for i, (label, count) in enumerate(label_counts.head(8).items()):
            plt.text(0.5, y_pos, f'{label}: {count}', 
                    transform=plt.gca().transAxes, ha='center', fontsize=10)
            y_pos -= 0.035
        
        plt.title('Visualization Info', fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计信息
        print("\nDetailed Label Distribution:")
        print(spots_to_show['label'].value_counts())
        
        return vis_image, label_colors

def get_transforms(mode: str = 'train'):
    """获取基本变换（无数据增强）"""
    # 返回None，因为我们将直接在_apply_transforms中处理变换
    return None

def create_dataloaders(data_root: str,
                      train_samples: List[str],
                      val_samples: Optional[List[str]] = None,
                      test_samples: Optional[List[str]] = None,
                      batch_size: int = 8,
                      patch_size: Optional[int] = 64,
                      max_spots_per_sample: int = 1000,
                      num_workers: int = 4,
                      auto_patch_size: bool = False,
                      skip_undetermined: bool = True,
                      validation_split: Optional[float] = None,
                      undersample_majority: bool = True,
                      undersample_ratio: float = 0.5) -> Dict[str, DataLoader]:
    """创建数据加载器
    
    Args:
        data_root: 数据根目录
        train_samples: 训练样本列表
        val_samples: 验证样本列表（可选，如果提供则直接使用）
        test_samples: 测试样本列表（可选）
        batch_size: 批次大小
        patch_size: patch大小
        max_spots_per_sample: 每个样本最大spot数量
        num_workers: 数据加载器工作进程数
        auto_patch_size: 是否启用自适应patch大小
        skip_undetermined: 是否跳过包含"undetermined"标签的样本
        validation_split: 验证集划分比例（如果提供则从训练数据中随机划分）
        undersample_majority: 是否对训练集中的多数类进行欠采样
        undersample_ratio: 欠采样比例，多数类保留的比例
    """
    
    def sample_contains_label(sample_id: str, target_label: str, data_root: str) -> bool:
        """
        检查样本中是否包含特定标签
        
        Args:
            sample_id: 样本ID
            target_label: 目标标签
            data_root: 数据根目录
            
        Returns:
            bool: 是否包含目标标签
        """
        try:
            # 加载标签文件
            label_dir = os.path.join(data_root, 'ST-pat', 'lbl')
            label_path = os.path.join(label_dir, f"{sample_id}_labeled_coordinates.tsv")
            
            if not os.path.exists(label_path):
                return False
                
            # 读取标签数据
            label_df = pd.read_csv(label_path, sep='\t')
            
            # 检查是否包含目标标签
            return target_label in label_df['label'].values
        except Exception as e:
            logging.warning(f"Error checking labels for sample {sample_id}: {str(e)}")
            return False
    
    def get_all_labels_in_sample(sample_id: str, data_root: str) -> set:
        """
        获取样本中包含的所有标签
        
        Args:
            sample_id: 样本ID
            data_root: 数据根目录
            
        Returns:
            set: 样本中包含的所有标签集合
        """
        try:
            # 加载标签文件
            label_dir = os.path.join(data_root, 'ST-pat', 'lbl')
            label_path = os.path.join(label_dir, f"{sample_id}_labeled_coordinates.tsv")
            
            if not os.path.exists(label_path):
                return set()
                
            # 读取标签数据
            label_df = pd.read_csv(label_path, sep='\t')
            
            # 返回所有标签的集合
            return set(label_df['label'].unique())
        except Exception as e:
            logging.warning(f"Error getting labels for sample {sample_id}: {str(e)}")
            return set()
    
    def get_all_available_labels(data_root: str, samples: List[str]) -> set:
        """
        获取所有样本中包含的所有标签
        
        Args:
            data_root: 数据根目录
            samples: 样本列表
            
        Returns:
            set: 所有样本中包含的所有标签集合
        """
        all_labels = set()
        for sample_id in samples:
            sample_labels = get_all_labels_in_sample(sample_id, data_root)
            all_labels.update(sample_labels)
        return all_labels
    
    # 如果提供了验证集划分比例，则从训练数据中随机划分验证集
    if validation_split is not None and validation_split > 0:
        # 获取所有可用的标签
        all_available_labels = get_all_available_labels(data_root, train_samples)
        logging.info(f"所有可用标签: {sorted(all_available_labels)}")
        
        # 从所有样本中收集所有spot数据
        all_spots = []
        spot_to_sample = []  # 记录每个spot属于哪个样本
        
        for sample_id in train_samples:
            try:
                # 加载标签文件
                label_dir = os.path.join(data_root, 'ST-pat', 'lbl')
                label_path = os.path.join(label_dir, f"{sample_id}_labeled_coordinates.tsv")
                
                if not os.path.exists(label_path):
                    logging.warning(f"标签文件不存在: {label_path}")
                    continue
                    
                # 读取标签数据
                label_df = pd.read_csv(label_path, sep='\t')
                
                # 过滤有效数据
                valid_spots = label_df.dropna(subset=['pixel_x', 'pixel_y', 'label'])
                
                # 存储spot数据和对应的样本ID
                for _, spot in valid_spots.iterrows():
                    all_spots.append((sample_id, spot['pixel_x'], spot['pixel_y'], spot['label']))
                    spot_to_sample.append(sample_id)
                
            except Exception as e:
                logging.error(f"加载样本{sample_id}的spot数据时出错: {str(e)}")
                continue
        
        # 计算验证集总大小
        total_val_size = max(1, int(len(all_spots) * validation_split))
        logging.info(f"总spot数量: {len(all_spots)}, 验证集spot数量: {total_val_size}")
        
        # 分离invasive cancer和非invasive cancer的spot
        invasive_cancer_spots = []
        non_invasive_cancer_spots = []
        
        for i, (sample_id, x, y, label) in enumerate(all_spots):
            if label == 'invasive cancer':
                invasive_cancer_spots.append(i)
            else:
                non_invasive_cancer_spots.append(i)
        
        # 计算验证集中应包含的invasive cancer spot数量（40%）
        invasive_val_count = int(total_val_size * 0.4)
        invasive_val_count = min(invasive_val_count, len(invasive_cancer_spots))
        
        # 从invasive cancer spot中随机选择
        if invasive_val_count > 0:
            selected_invasive_indices = random.sample(invasive_cancer_spots, invasive_val_count)
        else:
            selected_invasive_indices = []
        
        # 计算非invasive cancer spot的数量
        non_invasive_val_count = total_val_size - len(selected_invasive_indices)
        non_invasive_val_count = min(non_invasive_val_count, len(non_invasive_cancer_spots))
        
        # 从非invasive cancer spot中随机选择
        selected_non_invasive_indices = random.sample(non_invasive_cancer_spots, non_invasive_val_count)
        
        # 合并验证集spot索引
        val_spot_indices = selected_invasive_indices + selected_non_invasive_indices
        random.shuffle(val_spot_indices)
        
        # 确保验证集包含所有类别标签
        val_labels = set()
        missing_labels = set(all_available_labels)  # 初始设置为所有可用标签
        
        # 先检查已选择的验证集中是否包含所有标签
        for idx in val_spot_indices:
            _, _, _, label = all_spots[idx]
            val_labels.add(label)
        
        missing_labels = missing_labels - val_labels
        
        # 如果有缺失的标签，添加包含这些标签的spot
        for label in missing_labels:
            # 找到包含该标签且未被选中的spot
            candidates = []
            for i, (sample_id, x, y, spot_label) in enumerate(all_spots):
                if spot_label == label and i not in val_spot_indices:
                    candidates.append(i)
            
            if candidates:
                # 随机选择一个包含该标签的spot
                selected_idx = random.choice(candidates)
                val_spot_indices.append(selected_idx)
                
                # 确保验证集大小不超过总大小的1.2倍
                if len(val_spot_indices) > total_val_size * 1.2:
                    break
        
        # 调整验证集大小，确保比例
        if len(val_spot_indices) > total_val_size:
            # 优先移除非invasive cancer的spot
            to_remove = len(val_spot_indices) - total_val_size
            non_invasive_in_val = [idx for idx in val_spot_indices if all_spots[idx][3] != 'invasive cancer']
            
            if len(non_invasive_in_val) >= to_remove:
                # 随机移除非invasive cancer的spot
                remove_indices = random.sample(non_invasive_in_val, to_remove)
                val_spot_indices = [idx for idx in val_spot_indices if idx not in remove_indices]
            else:
                # 如果非invasive cancer的spot不够，移除所有非invasive cancer的spot，再移除invasive cancer的spot
                val_spot_indices = [idx for idx in val_spot_indices if all_spots[idx][3] == 'invasive cancer']
                remaining_to_remove = to_remove - len(non_invasive_in_val)
                if len(val_spot_indices) > remaining_to_remove:
                    val_spot_indices = val_spot_indices[:-remaining_to_remove]
        
        # 创建训练集和验证集样本列表
        # 记录哪些样本包含在验证集中
        val_sample_ids = set()
        train_sample_ids = set(train_samples)
        
        for idx in val_spot_indices:
            sample_id = all_spots[idx][0]
            val_sample_ids.add(sample_id)
        
        # 训练样本是所有没有被完全选为验证集的样本
        # 由于我们是按spot级别划分，所有原始样本都会保留在训练集中，但部分spot会被划分到验证集
        # 因此，这里我们仍然保留所有原始训练样本，但在创建数据集时会处理spot级别的划分
        train_samples_split = train_samples.copy()
        
        # 验证集样本是至少有一个spot被选中的样本
        val_samples_split = list(val_sample_ids)
        
        # 计算验证集统计信息
        val_spots_count = len(val_spot_indices)
        invasive_in_val_count = sum(1 for idx in val_spot_indices if all_spots[idx][3] == 'invasive cancer')
        invasive_ratio = invasive_in_val_count / val_spots_count if val_spots_count > 0 else 0
        
        # 更新验证集包含的标签
        val_labels = set()
        for idx in val_spot_indices:
            _, _, _, label = all_spots[idx]
            val_labels.add(label)
        
        logging.info(f"验证集包含的标签: {sorted(val_labels)}")
        logging.info(f"验证集样本数量: {len(val_samples_split)}")
        logging.info(f"验证集spot数量: {val_spots_count}")
        logging.info(f"训练集样本数量: {len(train_samples_split)}")
        logging.info(f"验证集中invasive cancer样本比例: {invasive_ratio:.2%}")
        
        # 保存验证集的spot索引，用于后续在数据集中应用
        import tempfile
        import pickle
        
        # 创建临时文件存储验证集spot索引
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump({
                'val_spot_indices': val_spot_indices,
                'all_spots': all_spots
            }, f)
            val_spots_file = f.name
        
        # 在训练和验证数据集创建时传递此文件路径
        train_dataset_kwargs = {'val_spots_file': None}  # 训练集不使用验证集spot索引
        val_dataset_kwargs = {'val_spots_file': val_spots_file}  # 验证集使用验证集spot索引
        
    else:
        
        # 最终验证
        final_val_labels = set()
        for sample in val_samples_split:
            sample_labels = get_all_labels_in_sample(sample, data_root)
            final_val_labels.update(sample_labels)
        
        final_invasive_in_val = sum(1 for s in val_samples_split if sample_contains_label(s, "invasive cancer", data_root))
        final_invasive_ratio = final_invasive_in_val / len(val_samples_split) if val_samples_split else 0
        
        logging.info(f"最终验证集包含的标签: {sorted(final_val_labels)}")
        logging.info(f"最终验证集样本数量: {len(val_samples_split)} (预期: {total_val_size})")
        logging.info(f"最终训练集样本数量: {len(train_samples_split)}")
        logging.info(f"最终验证集中invasive cancer样本比例: {final_invasive_ratio:.2%} (预期: 40%)")
        
        # 验证比例是否正确
        actual_ratio = len(val_samples_split) / len(all_train_samples)
        expected_ratio = validation_split
        logging.info(f"验证集实际比例: {actual_ratio:.2%} (配置比例: {expected_ratio:.2%})")
        
        # 如果验证集为空，则使用原始样本
        if not val_samples_split:
            logging.warning("验证集为空，使用原始样本划分")
            val_size = int(len(all_train_samples) * validation_split)
            val_samples_split = all_train_samples[-val_size:]
            train_samples_split = all_train_samples[:-val_size]
        # 使用提供的验证样本或默认验证样本
        train_samples_split = train_samples
        val_samples_split = val_samples
    
    # 创建训练数据集
    train_dataset = SpotFormerDataset(
        data_root=data_root,
        samples=train_samples_split,
        patch_size=patch_size if not auto_patch_size else None,
        max_spots_per_sample=max_spots_per_sample,
        mode='train',
        auto_patch_size=auto_patch_size,
        skip_undetermined=skip_undetermined,
        undersample_majority=undersample_majority,
        undersample_ratio=undersample_ratio,
        **(train_dataset_kwargs if validation_split is not None and validation_split > 0 else {})
    )
    
    # 获取训练集的标签映射
    train_label_to_idx = train_dataset.label_to_idx
    train_idx_to_label = train_dataset.idx_to_label
    
    # 创建验证数据集（如果提供了验证样本）
    if val_samples_split:
        # 创建临时数据集以获取所有标签
        temp_val_dataset = SpotFormerDataset(
            data_root=data_root,
            samples=val_samples_split,
            patch_size=patch_size if not auto_patch_size else None,
            max_spots_per_sample=max_spots_per_sample,
            mode='val',
            auto_patch_size=auto_patch_size,
            skip_undetermined=skip_undetermined,
            undersample_majority=False,  # 验证集不使用欠采样
            undersample_ratio=undersample_ratio
        )
        
        # 合并训练集和验证集的标签
        all_labels = set(train_label_to_idx.keys())
        for label in temp_val_dataset.label_to_idx.keys():
            all_labels.add(label)
        
        # 创建统一的标签映射
        sorted_labels = sorted(list(all_labels))
        unified_label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        unified_idx_to_label = {idx: label for label, idx in unified_label_to_idx.items()}
        
        # 更新训练集的标签映射
        train_dataset.label_to_idx = unified_label_to_idx
        train_dataset.idx_to_label = unified_idx_to_label
        train_dataset.num_classes = len(unified_label_to_idx)
        
        # 创建验证数据集，使用统一的标签映射
        val_dataset = SpotFormerDataset(
            data_root=data_root,
            samples=val_samples_split,
            patch_size=patch_size if not auto_patch_size else None,
            max_spots_per_sample=max_spots_per_sample,
            mode='val',
            auto_patch_size=auto_patch_size,
            skip_undetermined=skip_undetermined,
            undersample_majority=False,  # 验证集不使用欠采样
            undersample_ratio=undersample_ratio,
            **val_dataset_kwargs
        )
        
        # 更新验证集的标签映射
        val_dataset.label_to_idx = unified_label_to_idx
        val_dataset.idx_to_label = unified_idx_to_label
        val_dataset.num_classes = len(unified_label_to_idx)
        
        # 创建数据加载器
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=SpotFormerDataset.collate_fn,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=SpotFormerDataset.collate_fn,
                pin_memory=True
            )
        }
    else:
        # 只创建训练数据加载器
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=SpotFormerDataset.collate_fn,
                pin_memory=True
            )
        }
    
    # 测试集（可选）
    if test_samples:
        test_dataset = SpotFormerDataset(
            data_root=data_root,
            samples=test_samples,
            patch_size=patch_size if not auto_patch_size else None,
            max_spots_per_sample=max_spots_per_sample,
            mode='test',
            auto_patch_size=auto_patch_size,
            skip_undetermined=skip_undetermined,
            undersample_majority=False,  # 测试集不使用欠采样
            undersample_ratio=undersample_ratio
        )
        
        # 更新测试集的标签映射，使用与训练集相同的映射
        if val_samples_split:
            test_dataset.label_to_idx = unified_label_to_idx
            test_dataset.idx_to_label = unified_idx_to_label
            test_dataset.num_classes = len(unified_label_to_idx)
        else:
            test_dataset.label_to_idx = train_label_to_idx
            test_dataset.idx_to_label = train_idx_to_label
            test_dataset.num_classes = len(train_label_to_idx)
        
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SpotFormerDataset.collate_fn,
            pin_memory=True
        )
    
    # 返回统一后的标签映射
    if val_samples_split:
        return dataloaders, unified_label_to_idx, unified_idx_to_label
    else:
        return dataloaders, train_label_to_idx, train_idx_to_label

if __name__ == "__main__":
    # 测试SpotFormerDataset
    print("=" * 50)
    print("Testing SpotFormerDataset")
    print("=" * 50)
    
    # 使用较小的图像尺寸进行测试
    config = {
        'data': {
            'root_dir': 'd:/Project/python/SRNAsim/data',
            'samples': ['A1'],  # 只测试一个样本以加快速度
            'patch_size': 64,
            'auto_patch_size': True,
            'min_patch_size': 32,
            'max_patch_size': 128,
            'transforms': {
                'train': {
                    'resize': [224, 224],
                    'normalize': True
                },
                'val': {
                    'resize': [224, 224],
                    'normalize': True
                }
            }
        }
    }
    
    # 创建数据集
    dataset = SpotFormerDataset(
        data_root=config['data']['root_dir'],
        samples=config['data']['samples'],
        patch_size=config['data']['patch_size'],
        auto_patch_size=config['data']['auto_patch_size']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        if 'patches' in sample:
            print(f"Patches shape: {sample['patches'].shape}")
        if 'positions' in sample:
            print(f"Positions shape: {sample['positions'].shape}")
        if 'labels' in sample:
            print(f"Labels shape: {sample['labels'].shape}")
    
    print("\n" + "=" * 50)
    print("Visualizing sample spots (with star markers)")
    print("=" * 50)
    
    # 可视化样本中的所有spots，使用星形标记
    dataset.visualize_sample_spots(sample_idx=0, show_all_spots=True, show_patches=False, 
                                  spot_marker_size=15, use_star_marker=True)
    
    # 再次可视化，显示patches边界框
    print("\n" + "=" * 50)
    print("Visualizing sample spots (with patches)")
    print("=" * 50)
    dataset.visualize_sample_spots(sample_idx=0, show_all_spots=True, show_patches=True, 
                                  spot_marker_size=15, use_star_marker=True)
    
    # 使用圆形标记的可视化
    print("\n" + "=" * 50)
    print("Visualizing sample spots (with circle markers)")
    print("=" * 50)
    dataset.visualize_sample_spots(sample_idx=0, show_all_spots=True, show_patches=False, 
                                  spot_marker_size=12, use_star_marker=False)