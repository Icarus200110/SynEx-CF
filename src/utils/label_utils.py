import os
import pandas as pd
import numpy as np

def create_label_vocabulary(data_dir):
    """
    从所有标签文件中提取标签，创建词表
    
    Args:
        data_dir (str): 数据根目录
        
    Returns:
        dict: 标签到索引的映射
        dict: 索引到标签的映射
        int: 词表大小
    """
    label_dir = os.path.join(data_dir, 'ST-pat', 'lbl')
    all_labels = set()
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('_labeled_coordinates.tsv')]
    
    # 从每个文件中提取标签
    for label_file in label_files:
        file_path = os.path.join(label_dir, label_file)
        try:
            label_df = pd.read_csv(file_path, sep='\t')
            if 'label' in label_df.columns:
                file_labels = label_df['label'].unique()
                # 过滤掉"undetermined"标签
                filtered_labels = [label for label in file_labels if label != "undetermined"]
                all_labels.update(filtered_labels)
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
    
    # 创建词表
    all_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    vocab_size = len(all_labels)
    
    print(f"创建了包含 {vocab_size} 个标签的词表")
    print(f"标签类型: {all_labels}")
    
    return label_to_idx, idx_to_label, vocab_size

def get_neighbor_coordinates(x, y, grid_size=1):
    """
    获取给定坐标周围的相邻坐标
    
    Args:
        x (int): x坐标
        y (int): y坐标
        grid_size (int): 网格大小，默认为1
        
    Returns:
        list: 相邻坐标列表，包括中心点
    """
    neighbors = []
    for dx in range(-grid_size, grid_size + 1):
        for dy in range(-grid_size, grid_size + 1):
            neighbors.append((x + dx, y + dy))
    
    return neighbors

def create_spatial_context(spot_id, label_map, default_idx=0):
    """
    为给定的spot创建空间上下文
    
    Args:
        spot_id (str): spot ID，格式为'x坐标x y坐标'
        label_map (dict): 坐标到标签索引的映射
        default_idx (int): 默认标签索引，用于找不到标签的情况
        
    Returns:
        list: 包含中心点及其相邻点标签索引的列表
    """
    try:
        # 从spot ID中提取坐标
        x, y = map(int, spot_id.split('x'))
        
        # 获取相邻坐标
        neighbors = get_neighbor_coordinates(x, y)
        
        # 获取每个坐标的标签索引
        context = []
        for nx, ny in neighbors:
            neighbor_id = f"{nx}x{ny}"
            if neighbor_id in label_map:
                context.append(label_map[neighbor_id])
            else:
                context.append(default_idx)
        
        return context
    except Exception as e:
        print(f"为spot {spot_id} 创建空间上下文时出错: {e}")
        # 返回默认上下文
        return [default_idx] * 9  # 3x3网格有9个点