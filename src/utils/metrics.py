import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim

def mean_squared_error(y_true, y_pred):
    """
    计算均方误差
    
    Args:
        y_true (numpy.ndarray): 真实值
        y_pred (numpy.ndarray): 预测值
        
    Returns:
        float: 均方误差
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    计算均方根误差
    
    Args:
        y_true (numpy.ndarray): 真实值
        y_pred (numpy.ndarray): 预测值
        
    Returns:
        float: 均方根误差
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pearson_correlation(y_true, y_pred):
    """
    计算皮尔逊相关系数
    
    Args:
        y_true (numpy.ndarray): 真实值，形状为 (n_spots, n_genes)
        y_pred (numpy.ndarray): 预测值，形状为 (n_spots, n_genes)
        
    Returns:
        tuple: (spot_avg_corr, spot_correlations)
            - spot_avg_corr: 所有spot的平均皮尔逊相关系数
            - spot_correlations: 每个spot的皮尔逊相关系数列表
    """
    # 计算每个spot的相关系数
    spot_correlations = []
    for i in range(y_true.shape[0]):
        # 处理可能存在的NaN或无穷大值
        true_vals = y_true[i, :]
        pred_vals = y_pred[i, :]
        
        # 过滤掉NaN和无穷大值
        valid_indices = np.isfinite(true_vals) & np.isfinite(pred_vals)
        valid_true = true_vals[valid_indices]
        valid_pred = pred_vals[valid_indices]
        
        # 检查是否有足够的有效值且值有变化
        if len(valid_true) > 1 and np.var(valid_true) > 1e-8 and np.var(valid_pred) > 1e-8:
            corr, _ = pearsonr(valid_true, valid_pred)
            spot_correlations.append(corr)
        else:
            spot_correlations.append(np.nan)  # 对于无效的spot，添加NaN
    
    spot_avg_corr = np.nanmean(spot_correlations) if np.any(np.isfinite(spot_correlations)) else 0.0
    
    return spot_avg_corr, spot_correlations


def spearman_correlation(y_true, y_pred):
    """
    计算斯皮尔曼相关系数
    
    Args:
        y_true (numpy.ndarray): 真实值，形状为 (n_spots, n_genes)
        y_pred (numpy.ndarray): 预测值，形状为 (n_spots, n_genes)
        
    Returns:
        tuple: (spot_avg_corr, spot_correlations)
            - spot_avg_corr: 所有spot的平均斯皮尔曼相关系数
            - spot_correlations: 每个spot的斯皮尔曼相关系数列表
    """
    # 计算每个spot的相关系数
    spot_correlations = []
    for i in range(y_true.shape[0]):
        # 处理可能存在的NaN或无穷大值
        true_vals = y_true[i, :]
        pred_vals = y_pred[i, :]
        
        # 过滤掉NaN和无穷大值
        valid_indices = np.isfinite(true_vals) & np.isfinite(pred_vals)
        valid_true = true_vals[valid_indices]
        valid_pred = pred_vals[valid_indices]
        
        # 检查是否有足够的有效值且值有变化
        if len(valid_true) > 1 and np.var(valid_true) > 1e-8 and np.var(valid_pred) > 1e-8:
            corr, _ = spearmanr(valid_true, valid_pred)
            spot_correlations.append(corr)
        else:
            spot_correlations.append(np.nan)  # 对于无效的spot，添加NaN
    
    spot_avg_corr = np.nanmean(spot_correlations) if np.any(np.isfinite(spot_correlations)) else 0.0
    
    return spot_avg_corr, spot_correlations


def cosine_similarity_metric(y_true, y_pred):
    """
    计算余弦相似度指标
    
    Args:
        y_true (numpy.ndarray): 真实值，形状为 (n_spots, n_genes)
        y_pred (numpy.ndarray): 预测值，形状为 (n_spots, n_genes)
        
    Returns:
        tuple: (cosine_avg, cosine_spots)
            - cosine_avg: 所有spot的平均余弦相似度
            - cosine_spots: 每个spot的余弦相似度列表
    """
    cosine_spots = []
    
    for i in range(y_true.shape[0]):
        # 处理可能存在的NaN或无穷大值
        true_vals = y_true[i, :]
        pred_vals = y_pred[i, :]
        
        # 过滤掉NaN和无穷大值
        valid_indices = np.isfinite(true_vals) & np.isfinite(pred_vals)
        valid_true = true_vals[valid_indices]
        valid_pred = pred_vals[valid_indices]
        
        # 检查是否有足够的有效值
        if len(valid_true) > 1:
            # 计算余弦相似度
            dot_product = np.dot(valid_true, valid_pred)
            norm_true = np.linalg.norm(valid_true)
            norm_pred = np.linalg.norm(valid_pred)
            
            # 避免除零错误
            if norm_true > 0 and norm_pred > 0:
                cosine_sim = dot_product / (norm_true * norm_pred)
                cosine_spots.append(cosine_sim)
            else:
                cosine_spots.append(np.nan)
        else:
            cosine_spots.append(np.nan)  # 对于无效的spot，添加NaN
    
    cosine_avg = np.nanmean(cosine_spots) if np.any(np.isfinite(cosine_spots)) else 0.0
    
    return cosine_avg, cosine_spots


def explained_variance_score(y_true, y_pred):
    """
    计算解释方差得分
    
    Args:
        y_true (numpy.ndarray): 真实值，形状为 (n_spots, n_genes)
        y_pred (numpy.ndarray): 预测值，形状为 (n_spots, n_genes)
        
    Returns:
        tuple: (evs_avg, evs_spots)
            - evs_avg: 所有spot的平均解释方差得分
            - evs_spots: 每个spot的解释方差得分列表
    """
    evs_spots = []
    
    for i in range(y_true.shape[0]):
        # 处理可能存在的NaN或无穷大值
        true_vals = y_true[i, :]
        pred_vals = y_pred[i, :]
        
        # 过滤掉NaN和无穷大值
        valid_indices = np.isfinite(true_vals) & np.isfinite(pred_vals)
        valid_true = true_vals[valid_indices]
        valid_pred = pred_vals[valid_indices]
        
        # 检查是否有足够的有效值
        if len(valid_true) > 1:
            # 计算解释方差得分
            variance_true = np.var(valid_true)
            if variance_true > 0:
                evs = 1 - np.var(valid_true - valid_pred) / variance_true
                evs_spots.append(evs)
            else:
                evs_spots.append(np.nan)
        else:
            evs_spots.append(np.nan)  # 对于无效的spot，添加NaN
    
    evs_avg = np.nanmean(evs_spots) if np.any(np.isfinite(evs_spots)) else 0.0
    
    return evs_avg, evs_spots


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model (torch.nn.Module): 模型
        dataloader (torch.utils.data.DataLoader): 数据加载器
        device (torch.device): 设备
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    all_true = []
    all_pred = []
    
    # 保存每个样本的预测结果
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, (label_tensor, image_tensor, expressions) in enumerate(dataloader):
            # 检查数据中是否存在NaN或无穷大值
            if torch.isnan(label_tensor).any() or torch.isinf(label_tensor).any() or \
               torch.isnan(expressions).any() or torch.isinf(expressions).any():
                print("警告：检测到NaN或无穷大值，跳过此批次")
                continue
                
            label_tensor = label_tensor.to(device)
            image_tensor = image_tensor.to(device)
            expressions = expressions.to(device)
            
            # 前向传播
            outputs = model(label_tensor, image_tensor)
            
            # 检查模型输出是否包含NaN或无穷大值
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("警告：模型输出包含NaN或无穷大值，跳过此批次")
                continue
            
            # 收集真实值和预测值
            # 确保预测值为整数（计数数据）
            outputs_rounded = torch.round(outputs) if not model.training else outputs
            all_true.append(expressions.cpu().numpy())
            all_pred.append(outputs_rounded.cpu().numpy())
    
    # 合并批次数据
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    
    # 计算评估指标
    pearson_avg, pearson_spots = pearson_correlation(y_true, y_pred)
    spearman_avg, spearman_spots = spearman_correlation(y_true, y_pred)
    cosine_avg, cosine_spots = cosine_similarity_metric(y_true, y_pred)
    evs_avg, evs_spots = explained_variance_score(y_true, y_pred)
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'pearson': pearson_avg,
        'spearman': spearman_avg,
        'cosine_similarity': cosine_avg,
        'explained_variance': evs_avg,
        'mean_abs_error': np.mean(np.abs(y_true - y_pred)),  # 平均绝对误差
        'median_abs_error': np.median(np.abs(y_true - y_pred)),  # 中位数绝对误差
        'spot_metrics': [{
            'pearson': p, 
            'spearman': s,
            'cosine_similarity': cs,
            'explained_variance': ev,
            'mse': mean_squared_error(y_true[i:i+1], y_pred[i:i+1]),
            'rmse': root_mean_squared_error(y_true[i:i+1], y_pred[i:i+1])
        } for i, (p, s, cs, ev) in enumerate(zip(pearson_spots, spearman_spots, cosine_spots, evs_spots))]
    }
    
    # 添加计数数据特有的评估指标
    # 计算相对误差（对于非零真实值）
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        rel_errors = np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
        metrics['mean_rel_error'] = np.mean(rel_errors)  # 平均相对误差
    else:
        metrics['mean_rel_error'] = np.nan
    
    return metrics