import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PatchEmbed(nn.Module):
    """图像块嵌入模块，将图像分割为块并投影到嵌入空间"""
    
    def __init__(self, img_size: int = 64, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        """
        初始化图像块嵌入模块
        
        Args:
            img_size: 输入图像大小
            patch_size: 图像块大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
        """
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 使用卷积层实现图像块分割和嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, in_chans, img_size, img_size]
            
        Returns:
            嵌入后的图像块 [batch_size, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"输入图像大小({H}x{W})与期望大小({self.img_size}x{self.img_size})不匹配"
        
        # [B, C, H, W] -> [B, embed_dim, grid_size, grid_size]
        x = self.proj(x)
        # [B, embed_dim, grid_size, grid_size] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x


class Attention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        """
        初始化多头注意力机制
        
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数量
            qkv_bias: 是否在qkv线性层中使用偏置
            attn_drop: 注意力 dropout 概率
            proj_drop: 输出投影 dropout 概率
        """
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"特征维度({dim})必须能被注意力头数量({num_heads})整除"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # qkv 线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, num_patches, dim]
            
        Returns:
            注意力加权后的张量 [batch_size, num_patches, dim]
        """
        B, N, C = x.shape
        
        # [B, N, C] -> [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 分离 q, k, v
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim] * 3
        
        # 计算注意力分数
        # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 注意力加权和
        # [B, num_heads, N, N] @ [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop_rate: float = 0.):
        """
        初始化多层感知机
        
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度
            drop_rate: dropout 概率
        """
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, num_patches, in_features]
            
        Returns:
            输出张量 [batch_size, num_patches, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class Block(nn.Module):
    """Transformer块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, 
                 drop_rate: float = 0., attn_drop_rate: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        """
        初始化Transformer块
        
        Args:
            dim: 输入特征维度
            num_heads: 注意力头数量
            mlp_ratio: MLP隐藏层维度与输入维度的比值
            qkv_bias: 是否在qkv线性层中使用偏置
            drop_rate: dropout 概率
            attn_drop_rate: 注意力 dropout 概率
            act_layer: 激活函数
            norm_layer: 归一化层
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      out_features=dim, drop_rate=drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, num_patches, dim]
            
        Returns:
            输出张量 [batch_size, num_patches, dim]
        """
        # 残差连接 + 多头注意力
        x = x + self.attn(self.norm1(x))
        # 残差连接 + MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class SpotViT(nn.Module):
    """轻量化ViT模型，用于空间转录组学spot分类"""
    
    def __init__(self, num_classes: int, img_size: int = 64, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, **kwargs):
        # 从kwargs中获取img_size，保持与原有代码的兼容性
        if 'img_size' in kwargs:
            img_size = kwargs['img_size']
        # 原有代码中可能将输入图像大小命名为patch_size，这里需要特别处理
        if 'patch_size' in kwargs:
            # 注意：这里的patch_size实际上是输入图像的大小
            img_size = kwargs['patch_size']
        """
        初始化轻量化ViT模型
        
        Args:
            num_classes: 分类类别数
            img_size: 输入图像大小
            patch_size: 图像块大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            depth: Transformer块数量
            num_heads: 注意力头数量
            mlp_ratio: MLP隐藏层维度与输入维度的比值
            qkv_bias: 是否在qkv线性层中使用偏置
            drop_rate: dropout 概率
            attn_drop_rate: 注意力 dropout 概率
            norm_layer: 归一化层
        """
        super(SpotViT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 图像块嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # 保存输入图像大小，方便后续使用
        self.img_size = img_size
        num_patches = self.patch_embed.num_patches
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # 归一化层
        self.norm = norm_layer(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, patches: torch.Tensor, positions: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            patches: 输入图像patches [batch_size, num_spots, 3, patch_size, patch_size]
            positions: spot位置信息 [batch_size, num_spots, 2]（当前版本未使用）
            mask: 掩码信息 [batch_size, num_spots]（当前版本未使用）
            
        Returns:
            分类logits [batch_size, num_spots, num_classes]
        """
        batch_size, num_spots, channels, height, width = patches.shape
        
        # 重塑为 [batch_size * num_spots, channels, height, width]
        x = patches.view(-1, channels, height, width)
        
        # 图像块嵌入
        x = self.patch_embed(x)
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 经过Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        # 全局平均池化，只使用cls_token或平均所有patch
        # [B*num_spots, num_patches, embed_dim] -> [B*num_spots, embed_dim]
        x = x.mean(dim=1)
        
        # 分类头
        x = self.head(x)
        
        # 重塑回 [batch_size, num_spots, num_classes]
        x = x.view(batch_size, num_spots, -1)
        
        return x
    
    def predict_spots(self, patches: torch.Tensor, positions: Optional[torch.Tensor] = None, 
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        预测spot的类别概率
        
        Args:
            patches: 输入图像patches [batch_size, num_spots, 3, patch_size, patch_size]
            positions: spot位置信息 [batch_size, num_spots, 2]（当前版本未使用）
            mask: 掩码信息 [batch_size, num_spots]（当前版本未使用）
            
        Returns:
            预测概率 [batch_size, num_spots, num_classes]
        """
        logits = self.forward(patches, positions, mask)
        return F.softmax(logits, dim=-1)


# 轻量化ViT配置
def spotvit_tiny(num_classes: int, **kwargs) -> SpotViT:
    """构建轻量化ViT-Tiny模型"""
    return SpotViT(
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=2.,
        **kwargs
    )


def spotvit_small(num_classes: int, **kwargs) -> SpotViT:
    """构建轻量化ViT-Small模型"""
    return SpotViT(
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.,
        **kwargs
    )


def spotvit_base(num_classes: int, **kwargs) -> SpotViT:
    """构建轻量化ViT-Base模型"""
    return SpotViT(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        **kwargs
    )


def create_spotresnet(num_classes: int, model_type: str = 'spotvit_small', **kwargs) -> nn.Module:
    """
    创建模型的工厂函数，保持与原有接口兼容
    
    Args:
        num_classes: 分类类别数
        model_type: 模型类型，可选'spotvit_tiny', 'spotvit_small', 'spotvit_base'
        **kwargs: 其他参数
        
    Returns:
        构建好的轻量化ViT模型
    """
    if model_type == 'spotvit_tiny':
        return spotvit_tiny(num_classes, **kwargs)
    elif model_type == 'spotvit_small':
        return spotvit_small(num_classes, **kwargs)
    elif model_type == 'spotvit_base':
        return spotvit_base(num_classes, **kwargs)
    else:
        # 保持向后兼容，默认使用spotvit_small
        return spotvit_small(num_classes, **kwargs)


# 保持原有类名的别名，确保兼容性
SpotResNet = SpotViT


if __name__ == "__main__":
    print(create_spotresnet(6, 'spotvit_small'))