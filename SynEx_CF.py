import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.external.mamba_vision.mamba_vision import MambaVision
from models.external.ka_gnn.ka_gnn import KA_GNN, KAN_linear, NaiveFourierKANLayer

# 组织类型标签映射
LABEL2ID = {
    'invasive cancer': 0, 
    'breast glands': 1, 
    'immune infiltrate': 2, 
    'cancer in situ': 3, 
    'connective tissue': 4, 
    'adipose tissue': 5, 
    'undetermined': -1
}

# 组织类型数量（包括undetermined）
NUM_CLASSES = len(LABEL2ID)

class TissueTypeEmbedding(nn.Module):
    """
    组织类型嵌入模块，将3x3邻域的组织类型标签转换为嵌入向量
    """
    def __init__(self, embedding_dim=64):
        """
        初始化组织类型嵌入模块
        
        参数:
            embedding_dim: 嵌入向量维度
        """
        super(TissueTypeEmbedding, self).__init__()
        
        # 创建嵌入层，+1是因为undetermined标签为-1，我们将其映射到0
        self.embedding = nn.Embedding(NUM_CLASSES + 1, embedding_dim, padding_idx=0)
        
        # 3x3卷积层，用于聚合邻域嵌入特征
        self.conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU()
        
        # 全局平均池化，将3x3特征聚合为单个向量
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, neighbor_labels):
        """
        前向传播
        
        参数:
            neighbor_labels: 3x3邻域组织类型标签，形状为 [B, N, 3, 3]
            B: 批次大小
            N: spot数量
        
        返回:
            组织类型嵌入特征，形状为 [B, N, embedding_dim]
        """
        B, N, H, W = neighbor_labels.shape
        
        # 将-1（undetermined）映射到0
        neighbor_labels = neighbor_labels.clone()
        neighbor_labels[neighbor_labels == -1] = 0
        
        # 嵌入每个组织类型标签
        # [B, N, H, W] -> [B*N, H, W]
        neighbor_labels = neighbor_labels.view(B*N, H, W)
        # [B*N, H, W] -> [B*N, H, W, embedding_dim]
        embedded = self.embedding(neighbor_labels)
        # [B*N, H, W, embedding_dim] -> [B*N, embedding_dim, H, W]
        embedded = embedded.permute(0, 3, 1, 2)
        
        # 卷积聚合邻域特征
        # [B*N, embedding_dim, H, W] -> [B*N, embedding_dim, H, W]
        conv_out = self.relu(self.bn(self.conv(embedded)))
        
        # 全局平均池化
        # [B*N, embedding_dim, H, W] -> [B*N, embedding_dim, 1, 1]
        pooled = self.global_avg_pool(conv_out)
        # [B*N, embedding_dim, 1, 1] -> [B*N, embedding_dim]
        pooled = pooled.view(B*N, -1)
        # [B*N, embedding_dim] -> [B, N, embedding_dim]
        pooled = pooled.view(B, N, -1)
        
        return pooled

class SpatialContextKAGNN(nn.Module):
    """
    KA-GNN空间上下文特征提取模块
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=128, num_layers=2, grid_feat=5):
        """
        初始化KA-GNN空间上下文特征提取模块
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度（减少为32，降低参数量）
            output_dim: 输出特征维度（减少为128，降低参数量）
            num_layers: 层数（保持2层）
            grid_feat: 网格特征维度（减少为5，降低参数量）
        """
        super(SpatialContextKAGNN, self).__init__()
        
        # 输入层：使用KAN线性层
        self.kan_line = KAN_linear(input_dim, hidden_dim, grid_feat, addbias=False)
        
        # 中间层：使用NaiveFourierKANLayer
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_dim, hidden_dim, grid_feat, addbias=False))
        
        self.linear_out = KAN_linear(hidden_dim, output_dim, grid_feat, addbias=False)
        
        # 激活函数和归一化
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征，形状为 [N, input_dim]
            edge_index: 边索引，形状为 [2, E]
        
        返回:
            空间上下文特征，形状为 [N, output_dim]
        """
        # 使用DGL图结构
        import dgl
        
        # 将PyG edge_index转换为DGL图
        g = dgl.graph((edge_index[0], edge_index[1]))
        
        # 输入层
        h = self.kan_line(x)
        
        # 中间层
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            h = self.relu(h)
        
        # 输出层：简化为单个线性层
        h = self.linear_out(h)
        
        # 归一化
        h = self.bn(h)
        
        return h

class MambaVisionImageFeatureExtractor(nn.Module):
    """
    MambaVision图像特征提取模块
    """
    def __init__(self, output_dim=512, freeze_mamba=False, freeze_layers=11):
        """
        初始化MambaVision图像特征提取模块
        
        参数:
            output_dim: 输出特征维度（增加为512，增强图像特征表示）
            freeze_mamba: 是否冻结MambaVision参数
            freeze_layers: 冻结的层数
        """
        super(MambaVisionImageFeatureExtractor, self).__init__()
        
        # 加载MambaVision模型，使用正确的参数配置
        try:
            print("Loading MambaVision model with correct configuration...")
            # 使用tiny模型的参数配置，根据MambaVision类定义调整参数名
            self.mamba_vision = MambaVision(
                dim=80,  # 基础维度
                in_dim=32,  # 输入维度
                depths=[1, 3, 8, 4],  # 每个阶段的层数
                window_size=[8, 8, 14, 7],  # 每个阶段的窗口大小
                mlp_ratio=4,  # MLP比率
                num_heads=[2,4,8,16],
                in_chans=3,  # 输入通道数
                num_classes=output_dim,  # 输出类别数
                drop_path_rate=0.2,  # drop path比率
            )
            print("MambaVision model loaded successfully")
        except Exception as e:
            print(f"Error loading MambaVision: {e}")
            # 如果加载失败，使用更复杂的简化版本
            print("Using enhanced simplified MambaVision implementation")
            self.mamba_vision = nn.Sequential(
                # 增加卷积层深度和宽度
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # 增加全连接层
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, output_dim)
            )
        
        # 激活函数和归一化
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        
        # 添加上采样层，将112x112的patch调整为224x224
        #self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # 冻结MambaVision参数
        if freeze_mamba:
            self.freeze_mamba_parameters(freeze_layers)
    
    def freeze_mamba_parameters(self, freeze_layers):
        """
        冻结MambaVision的参数，只训练最后几层和分类头
        
        参数:
            freeze_layers: 冻结的层数
        """
        # 冻结所有参数
        for param in self.mamba_vision.parameters():
            param.requires_grad = False
        
        # 解冻更多层（从4个参数组增加到8个）
        for param in list(self.mamba_vision.parameters())[-8:]:
            param.requires_grad = True
        
        print(f"Frozen MambaVision parameters, training the last 8 parameter groups.")
    
    def forward(self, patches):
        """
        前向传播
        
        参数:
            patches: 图像patch，形状为 [B*N, C, H, W]
            B: 批次大小
            N: spot数量
            C: 通道数
            H: 高度
            W: 宽度
        
        返回:
            图像特征，形状为 [B*N, output_dim]
        """
        # 将112x112的patch调整为224x224
        # [B*N, C, 112, 112] -> [B*N, C, 224, 224]
        #patches = self.upsample(patches)
        
        # 使用MambaVision提取特征
        # [B*N, C, 224, 224] -> [B*N, output_dim]
        features = self.mamba_vision(patches)
        features = self.relu(features)
        features = self.bn(features)
        
        # 直接返回B*N的特征，由调用者负责重塑
        return features

class SSDFNet(nn.Module):
    """
    Spatial-Semantic Dynamic Fusion Network (SSDF-Net)
    创新的特征融合模块，结合空间结构和语义信息
    """
    def __init__(self, mask_dim, image_dim, fusion_dim=512, hidden_dim=256):
        """
        初始化SSDF-Net特征融合模块
        
        参数:
            mask_dim: 掩码/空间特征维度
            image_dim: 图像特征维度
            fusion_dim: 融合特征维度
            hidden_dim: 隐藏层维度
        """
        super(SSDFNet, self).__init__()
        self.fusion_dim = fusion_dim
        # ------------------- 1. 掩码引导变形对齐模块 -------------------
        self.mask_embedding = nn.Linear(mask_dim, fusion_dim)
        self.image_embedding = nn.Linear(image_dim, fusion_dim)
        
        # 轻量级空间变形网络
        self.deform_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim * 2)  # 输出变形偏移量和权重
        )
        
        # ------------------- 2. 跨模态动态超图构建模块 -------------------
        # 超图节点嵌入
        self.node_embedding = nn.Linear(fusion_dim * 2, fusion_dim)
        
        # 超边权重计算
        self.hyperedge_weight = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 双曲空间投影
        self.hyperbolic_proj = nn.Linear(fusion_dim, fusion_dim)
        
        # ------------------- 3. 双路径聚合与门控输出模块 -------------------
        # 路径A：语义增强路径
        self.semantic_path = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fusion_dim)
        )
        
        # 路径B：细节保留路径
        self.detail_path = nn.Identity()  # 残差连接
        
        # 自适应门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的门控权重
        )
        
        # 最终输出层
        self.output_layer = nn.Linear(fusion_dim, fusion_dim)
        
        # 激活函数和归一化
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(fusion_dim)
    
    def forward(self, mask_features, image_features):
        """
        前向传播
        
        参数:
            mask_features: 掩码/空间上下文特征，形状为 [B, N, mask_dim]
            image_features: 图像特征，形状为 [B, N, image_dim]
        
        返回:
            融合特征，形状为 [B, N, fusion_dim]
        """
        B, N, _ = mask_features.shape
        
        # ------------------- 1. 掩码引导变形对齐模块 -------------------
        # 映射到相同维度
        # [B, N, mask_dim] -> [B, N, fusion_dim]
        mask_emb = self.relu(self.mask_embedding(mask_features))
        # [B, N, image_dim] -> [B, N, fusion_dim]
        image_emb = self.relu(self.image_embedding(image_features))
        
        # 计算变形偏移量
        # [B, N, fusion_dim*2] -> [B*N, fusion_dim*2]
        combined_emb = torch.cat([mask_emb, image_emb], dim=-1).view(B*N, -1)
        # [B*N, fusion_dim*2] -> [B*N, fusion_dim*2]
        deform_params = self.deform_net(combined_emb)
        offset, weight = torch.split(deform_params, self.fusion_dim, dim=-1)
        
        # 应用变形到图像特征
        # [B*N, fusion_dim] -> [B*N, fusion_dim]
        deformed_image = image_emb.view(B*N, -1) + offset * torch.sigmoid(weight)
        deformed_image = deformed_image.view(B, N, -1)
        
        # ------------------- 2. 跨模态动态超图构建模块 -------------------
        # 构建节点特征：拼接变形后的图像特征和掩码特征
        # [B, N, fusion_dim*2]
        node_features = torch.cat([deformed_image, mask_emb], dim=-1)
        # [B*N, fusion_dim*2] -> [B*N, fusion_dim]
        node_emb = self.relu(self.node_embedding(node_features.view(B*N, -1)))
        
        # 双曲空间投影
        # [B*N, fusion_dim] -> [B*N, fusion_dim]
        hyperbolic_node = self.relu(self.hyperbolic_proj(node_emb))
        
        # 构建动态超边（基于空间邻接和语义相似性）
        # 这里简化实现，使用KNN构建超边
        hyperbolic_node = hyperbolic_node.view(B, N, -1)
        
        # 计算节点间相似度
        # [B, N, fusion_dim] @ [B, fusion_dim, N] -> [B, N, N]
        similarity = torch.matmul(hyperbolic_node, hyperbolic_node.transpose(1, 2))
        
        # 选择K个最近邻构建超边
        K = 5
        _, topk_indices = torch.topk(similarity, K, dim=-1)
        
        # ------------------- 3. 双路径聚合与门控输出模块 -------------------
        # 路径A：语义增强路径 - 超图推理
        semantic_enhanced = []
        for b in range(B):
            batch_node = hyperbolic_node[b]
            batch_topk = topk_indices[b]
            
            # 超图聚合
            aggregated = []
            for i in range(N):
                # 获取第i个节点的K个邻居
                neighbors = batch_node[batch_topk[i]]
                # 计算邻居特征的加权和
                node_similarity = similarity[b, i, batch_topk[i]]
                weights = torch.softmax(node_similarity, dim=-1)
                aggregated.append(torch.sum(neighbors * weights.unsqueeze(-1), dim=0))
            
            semantic_enhanced.append(torch.stack(aggregated))
        
        # [B, N, fusion_dim]
        semantic_enhanced = torch.stack(semantic_enhanced)
        # [B*N, fusion_dim] -> [B*N, fusion_dim]
        semantic_enhanced = self.semantic_path(semantic_enhanced.view(B*N, -1)).view(B, N, -1)
        
        # 路径B：细节保留路径 - 残差连接
        # [B, N, fusion_dim]
        detail_preserved = self.detail_path(deformed_image)
        
        # 自适应门控：根据语义置信度动态融合两条路径
        # 计算门控权重
        # [B, N, fusion_dim*2] -> [B*N, fusion_dim*2] -> [B*N, 1] -> [B, N, 1]
        gate_input = torch.cat([semantic_enhanced, detail_preserved], dim=-1)
        gate_weight = self.gate_net(gate_input.view(B*N, -1)).view(B, N, 1)
        
        # 融合两条路径
        # [B, N, fusion_dim]
        fused = gate_weight * semantic_enhanced + (1 - gate_weight) * detail_preserved
        
        # 最终输出
        # [B*N, fusion_dim] -> [B*N, fusion_dim] -> [B, N, fusion_dim]
        fused = self.output_layer(fused.view(B*N, -1))
        fused = self.bn(fused)
        fused = self.relu(fused)
        fused = fused.view(B, N, -1)
        
        return fused

class GenePredictor(nn.Module):
    """
    基因预测模块，从融合特征预测基因表达量
    """
    def __init__(self, input_dim, n_genes=785, hidden_dim=256):
        """
        初始化基因预测模块
        
        参数:
            input_dim: 输入特征维度
            n_genes: 基因数量
            hidden_dim: 隐藏层维度
        """
        super(GenePredictor, self).__init__()
        
        # 预测网络
        self.predict_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_genes)
        )
    
    def forward(self, fused_features):
        """
        前向传播
        
        参数:
            fused_features: 融合特征，形状为 [B, N, input_dim]
        
        返回:
            预测的基因表达量，形状为 [B, N, n_genes]
        """
        B, N, _ = fused_features.shape
        
        # [B, N, input_dim] -> [B*N, input_dim]
        fused_features = fused_features.view(B*N, -1)
        
        # 预测基因表达量
        # [B*N, input_dim] -> [B*N, n_genes]
        gene_pred = self.predict_net(fused_features)
        
        # 重塑形状
        # [B*N, n_genes] -> [B, N, n_genes]
        gene_pred = gene_pred.view(B, N, -1)
        
        return gene_pred

class SynEx_CF(pl.LightningModule):
    """
    SynEx_CF模型，整合组织类型嵌入、GCN空间上下文特征提取、ViT图像特征提取、特征融合和基因预测
    """
    def __init__(self, n_genes=785, learning_rate=1e-5, label=None, freeze_vit=False, freeze_vit_layers=11):
        """
        初始化SynEx_CF模型
        
        参数:
            n_genes: 基因数量
            learning_rate: 学习率
            label: 标签数据，用于验证步骤中的聚类评估
            freeze_vit: 是否冻结ViT参数
            freeze_vit_layers: 冻结的ViT层数
        """
        super(SynEx_CF, self).__init__()
        
        # 组织类型嵌入模块
        self.tissue_embedding = TissueTypeEmbedding(embedding_dim=64)
        
        # KA-GNN空间上下文特征提取模块（减少参数量）
        self.ka_gnn = SpatialContextKAGNN(input_dim=64, hidden_dim=32, output_dim=128, num_layers=2, grid_feat=5)
        
        # MambaVision图像特征提取模块（减少参数量）
        self.mamba_vision = MambaVisionImageFeatureExtractor(output_dim=256, freeze_mamba=freeze_vit, freeze_layers=freeze_vit_layers)
        
        # SSDFNet特征融合模块（替换原有融合模块）
        self.fusion = SSDFNet(mask_dim=128, image_dim=256, fusion_dim=256, hidden_dim=128)
        
        # 基因预测模块
        self.gene_predictor = GenePredictor(input_dim=256, n_genes=n_genes, hidden_dim=128)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # 其他参数
        self.learning_rate = learning_rate
        self.label = label
        self.n_genes = n_genes
    
    def get_neighbor_labels(self, labels, loc):
        """
        获取每个spot的3x3邻域组织类型标签
        
        参数:
            labels: 组织类型标签，形状为 [N]
            loc: spot坐标，形状为 [N, 2]
        
        返回:
            3x3邻域组织类型标签，形状为 [1, N, 3, 3]
        """
        N = labels.shape[0]
        
        # 初始化邻域标签
        neighbor_labels = torch.full((1, N, 3, 3), -1, dtype=torch.long, device=labels.device)
        
        # 将loc转换为整数坐标
        loc = loc.long()
        
        # 构建坐标到索引的映射
        coord_to_idx = {tuple(coord.tolist()): i for i, coord in enumerate(loc)}
        
        # 遍历每个spot
        for i in range(N):
            x, y = loc[i]
            
            # 遍历3x3邻域
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    
                    # 检查邻域坐标是否存在
                    if (nx, ny) in coord_to_idx:
                        neighbor_idx = coord_to_idx[(nx, ny)]
                        # 计算邻域位置
                        px, py = dx + 1, dy + 1  # 将[-1, 0, 1]映射到[0, 1, 2]
                        neighbor_labels[0, i, px, py] = labels[neighbor_idx]
        
        return neighbor_labels
    
    def forward(self, patches, loc, labels, adj):
        """
        前向传播
        
        参数:
            patches: 图像patch，形状为 [B, N, C, H, W]
            loc: spot坐标，形状为 [B, N, 2]
            labels: 组织类型标签，形状为 [B, N]
            adj: 邻接矩阵，形状为 [B, N, N]
        
        返回:
            预测的基因表达量，形状为 [B, N, n_genes]
        """
        B, N, C, H, W = patches.shape
        B_N = B * N  # 总spot数量
        
        # 获取3x3邻域组织类型标签
        neighbor_labels = self.get_neighbor_labels(labels.squeeze(0), loc.squeeze(0))
        
        # 组织类型嵌入
        # [1, N, 3, 3] -> [1, N, embedding_dim]
        tissue_features = self.tissue_embedding(neighbor_labels)
        
        # 扩展组织特征到批次大小
        # [1, N, embedding_dim] -> [B, N, embedding_dim]
        tissue_features = tissue_features.expand(B, N, -1)
        
        # KA-GNN空间上下文特征提取
        # 构建边索引 - 目前只支持单批次，因为DGL图不支持批次处理
        # 如果批次大小大于1，我们需要为每个样本单独处理
        spatial_features_list = []
        for b in range(B):
            # 获取当前样本的邻接矩阵
            adj_b = adj[b]
            # 构建边索引
            edge_index_b = adj_b.nonzero().t()
            # 获取当前样本的组织特征
            tissue_features_b = tissue_features[b]
            # [N, embedding_dim] -> [N, spatial_dim]
            spatial_features_b = self.ka_gnn(tissue_features_b, edge_index_b)
            # 添加到列表
            spatial_features_list.append(spatial_features_b)
        
        # 拼接所有样本的空间特征
        # [B, N, spatial_dim]
        spatial_features = torch.stack(spatial_features_list, dim=0)
        
        # MambaVision图像特征提取
        # [B, N, C, H, W] -> [B*N, C, H, W]
        patches_flat = patches.view(B*N, C, H, W)
        # [B*N, C, H, W] -> [B*N, image_dim]
        image_features_flat = self.mamba_vision(patches_flat)
        # [B*N, image_dim] -> [B, N, image_dim]
        image_features = image_features_flat.view(B, N, -1)
        
        # 特征融合
        # [B, N, spatial_dim] + [B, N, image_dim] -> [B, N, fusion_dim]
        fused_features = self.fusion(spatial_features, image_features)
        
        # 基因预测
        # [B, N, fusion_dim] -> [B, N, n_genes]
        gene_pred = self.gene_predictor(fused_features)
        
        return gene_pred
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤
        """
        patches, positions, exp, adj, *_, centers = batch
        
        # 获取组织类型标签
        # 注意：这里需要根据实际数据加载情况调整
        # 假设labels是batch中的第6个元素
        labels = None
        if len(batch) > 6:
            labels = batch[6]
        else:
            # 如果batch中没有labels，使用默认值
            labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=patches.device)
        
        # 前向传播
        pred = self(patches, positions, labels, adj)
        
        # 计算损失
        loss = self.mse_loss(pred.squeeze(0), exp.squeeze(0))
        
        # 计算PCC
        pred_np = pred.squeeze(0).cpu().detach().numpy()
        exp_np = exp.squeeze(0).cpu().numpy()
        pcc = self.calculate_pcc(pred_np, exp_np)
        
        # 记录损失和PCC，同时记录step和epoch指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def calculate_pcc(self, pred, exp):
        """
        计算预测值和真实值之间的皮尔逊相关系数（PCC）
        
        参数:
            pred: 预测的基因表达量，形状为 [N, G]
            exp: 真实的基因表达量，形状为 [N, G]
            N: spot数量
            G: 基因数量
        
        返回:
            平均PCC值
        """
        from scipy.stats import pearsonr
        import numpy as np
        
        # 计算每个基因的PCC
        pcc_list = []
        for g in range(pred.shape[1]):
            r, _ = pearsonr(pred[:, g], exp[:, g])
            pcc_list.append(r)
        
        # 计算平均PCC，排除NaN值
        pcc_array = np.array(pcc_list)
        avg_pcc = np.nanmean(pcc_array)
        
        return avg_pcc
    
    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        """
        patches, positions, exp, adj, *_, centers = batch
        
        # 获取组织类型标签
        labels = None
        if len(batch) > 6:
            labels = batch[6]
        else:
            labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=patches.device)
        
        # 前向传播
        pred = self(patches, positions, labels, adj)
        
        # 计算损失
        loss = self.mse_loss(pred.squeeze(0), exp.squeeze(0))
        
        # 计算PCC
        pred_np = pred.squeeze(0).cpu().detach().numpy()
        exp_np = exp.squeeze(0).cpu().numpy()
        pcc = self.calculate_pcc(pred_np, exp_np)
        
        # 记录损失和PCC，同时记录step和epoch指标
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        测试步骤
        """
        patches, positions, exp, adj, *_, centers = batch
        
        # 获取组织类型标签
        labels = None
        if len(batch) > 6:
            labels = batch[6]
        else:
            labels = torch.full((1, patches.shape[1]), -1, dtype=torch.long, device=patches.device)
        
        # 前向传播
        pred = self(patches, positions, labels, adj)
        
        # 计算损失
        loss = self.mse_loss(pred.squeeze(0), exp.squeeze(0))
        
        # 计算PCC
        pred_np = pred.squeeze(0).cpu().detach().numpy()
        exp_np = exp.squeeze(0).cpu().numpy()
        pcc = self.calculate_pcc(pred_np, exp_np)
        
        # 记录损失和PCC，同时记录step和epoch指标
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        配置优化器
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

if __name__ == "__main__":
    # 测试模型
    model = SynEx_CF(n_genes=785)
    
    # 创建测试数据
    B = 1
    N = 100
    C = 3
    H = 224
    W = 224
    
    patches = torch.randn(B, N, C, H, W)
    loc = torch.randint(0, 100, (B, N, 2))
    labels = torch.randint(0, 6, (B, N))
    adj = torch.eye(N).unsqueeze(0)
    
    # 前向传播
    pred = model(patches, loc, labels, adj)
    
    print(f"Input shape: {patches.shape}")
    print(f"Output shape: {pred.shape}")