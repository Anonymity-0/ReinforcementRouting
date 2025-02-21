import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPPOActor(nn.Module):
    """MAPPO Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: 动作概率分布 [batch_size, action_dim]
        """
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

class MAPPOCritic(nn.Module):
    """MAPPO Critic网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态值
        )
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            torch.Tensor: 状态值 [batch_size, 1]
        """
        return self.net(state)

class CentralizedMAPPOCritic(nn.Module):
    """MAPPO中心化Critic网络"""
    
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int):
        """
        初始化中心化Critic网络
        
        Args:
            state_dim: 每个智能体的状态维度
            num_agents: 智能体数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 计算全局状态维度
        global_state_dim = state_dim * num_agents
        
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)  # 为每个智能体输出一个状态值
        )
        
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            global_state: 全局状态张量 [batch_size, num_agents * state_dim]
            
        Returns:
            torch.Tensor: 每个智能体的状态值 [batch_size, num_agents]
        """
        return self.net(global_state) 