import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

class MaskedLinear(nn.Module):
    """带掩码的线性层"""
    
    def __init__(self, in_features: int, out_features: int):
        """
        初始化带掩码的线性层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 创建权重和偏置
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.orthogonal_(self.weight, gain=0.01)
        nn.init.constant_(self.bias, 0.0)
    
    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 输入张量 [batch_size, in_features]
            mask: 动作掩码 [batch_size, out_features]，True表示有效动作
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, out_features]
        """
        # 计算线性变换
        output = F.linear(input, self.weight, self.bias)
        
        # 应用掩码
        if mask is not None:
            output = torch.where(mask, output, torch.tensor(float('-inf'), device=output.device))
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        多头注意力层
        
        Args:
            d_model: 输入维度
            num_heads: 注意力头数
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        batch_size = x.size(0)
        
        # 线性变换
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)  # [batch_size, seq_len, d_model]
        
        # 分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # [batch_size, num_heads, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用注意力
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]
        
        # 输出线性变换
        output = self.W_o(context)  # [batch_size, seq_len, d_model]
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, out_channels]
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = F.relu(out)
        
        return out

class PPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        PPO策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            ResidualBlock(state_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # 注意力层
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: logits张量 [batch_size, action_dim]
        """
        # 特征提取
        features = self.feature_net(state)  # [batch_size, hidden_dim]
        
        # 注意力处理
        features = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        features = self.attention(features)  # [batch_size, 1, hidden_dim]
        features = features.squeeze(1)  # [batch_size, hidden_dim]
        
        # 策略输出
        logits = self.policy_head(features)  # [batch_size, action_dim]
        
        # 使用温度参数调节logits
        scaled_logits = logits / self.temperature
        
        return scaled_logits
    
    def get_action_distribution(self, state: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
        """
        获取动作概率分布
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            valid_actions: 有效动作列表
            
        Returns:
            torch.Tensor: 动作概率分布 [batch_size, action_dim]
        """
        # 获取logits
        logits = self.forward(state)  # [batch_size, action_dim]
        
        # 创建mask，将无效动作的logits设为负无穷
        mask = torch.full_like(logits, float('-inf'))
        valid_indices = torch.tensor(valid_actions, device=state.device)
        mask[..., valid_indices] = 0
        
        # 应用mask并计算softmax
        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)
        
        return probs

class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        PPO价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            ResidualBlock(state_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # 注意力层
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: 价值张量 [batch_size, 1]
        """
        # 特征提取
        features = self.feature_net(state)  # [batch_size, hidden_dim]
        
        # 注意力处理
        features = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        features = self.attention(features)  # [batch_size, 1, hidden_dim]
        features = features.squeeze(1)  # [batch_size, hidden_dim]
        
        # 价值输出
        value = self.value_head(features)  # [batch_size, 1]
        
        return value 