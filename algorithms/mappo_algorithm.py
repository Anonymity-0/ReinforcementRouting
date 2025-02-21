import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from .routing_interface import RoutingAlgorithm
from .networks.mappo_nets import MAPPOActor, MAPPOCritic, CentralizedMAPPOCritic

class MAPPOAlgorithm(RoutingAlgorithm):
    """MAPPO算法实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MAPPO算法
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        
        # 获取网络参数
        self.total_satellites = config['total_satellites']
        self.max_buffer_size = config['max_buffer_size']
        self.max_queue_length = config['max_queue_length']
        
        # 计算状态维度
        # 状态包括:
        # 1. 当前节点特征 (3 + 3 + 1 = 7)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        # 2. 目标节点特征 (3 + 3 + 1 = 7)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        # 3. 链路状态 (total_satellites)
        # 4. 全局特征 (2)
        #    - 当前节点ID (1)
        #    - 目标节点ID (1)
        self.state_dim = 7 + 7 + self.total_satellites + 2
        
        # 动作维度为卫星数量
        self.action_dim = self.total_satellites
        
        # 获取MAPPO参数
        self.device = torch.device(config.get('device', 'cpu'))
        self.hidden_dim = config.get('hidden_dim', 256)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_param = config.get('clip_param', 0.2)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.use_centralized_critic = config.get('use_centralized_critic', True)
        
        # 初始化网络
        self.actor = MAPPOActor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        if self.use_centralized_critic:
            self.critic = CentralizedMAPPOCritic(self.state_dim, self.total_satellites, self.hidden_dim).to(self.device)
        else:
            self.critic = MAPPOCritic(self.state_dim, self.hidden_dim).to(self.device)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # 初始化经验回放缓冲区
        self.buffer = []
        
        # 训练模式
        self.training = True
    
    def get_next_hop(self, current_node: int, target_node: int, state: Dict[str, Any]) -> int:
        """
        获取下一跳节点
        
        Args:
            current_node: 当前节点
            target_node: 目标节点
            state: 状态信息
            
        Returns:
            int: 下一跳节点ID
        """
        # 如果已到达目标节点
        if current_node == target_node:
            return current_node
            
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        
        # 获取有效动作
        valid_actions = state['topology']['object'].get_valid_neighbors(current_node)
        if not valid_actions:
            return -1
        
        # 获取动作概率分布
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            
            # 只保留有效动作的概率
            valid_probs = action_probs[0, valid_actions]
            valid_probs = valid_probs / valid_probs.sum()  # 重新归一化
            
            if not self.training:
                # 评估模式：选择概率最高的动作
                action_idx = valid_probs.argmax().item()
            else:
                # 训练模式：从分布中采样动作
                dist = torch.distributions.Categorical(valid_probs)
                action_idx = dist.sample().item()
            
            next_hop = valid_actions[action_idx]
        
        return next_hop
    
    def update(self, states: List[Dict], actions: np.ndarray, rewards: np.ndarray,
             next_states: List[Dict], dones: np.ndarray) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Args:
            states: 状态列表
            actions: 动作数组
            rewards: 奖励数组
            next_states: 下一个状态列表
            dones: 结束标志数组
            
        Returns:
            Dict[str, float]: 训练信息
        """
        if not self.training:
            return {}
            
        # 将数据添加到缓冲区
        self.buffer.extend(zip(states, actions, rewards, next_states, dones))
        
        # 如果缓冲区达到一定大小，进行更新
        if len(self.buffer) >= self.batch_size:
            return self._update_networks()
            
        return {}
    
    def _update_networks(self) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Returns:
            Dict[str, float]: 训练信息
        """
        # 从缓冲区中采样数据
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        
        # 预处理数据
        states_tensor = torch.cat([self._preprocess_state(s) for s in states])
        next_states_tensor = torch.cat([self._preprocess_state(s) for s in next_states])
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(states_tensor)
            next_values = self.critic(next_states_tensor)
            
            # 计算TD误差
            deltas = rewards_tensor + self.gamma * next_values * (1 - dones_tensor) - values
            
            # 计算GAE优势
            advantages = torch.zeros_like(deltas)
            advantage = 0.0
            for t in reversed(range(len(self.buffer))):
                advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * advantage
                advantages[t] = advantage
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算目标值
            returns = advantages + values
        
        # 更新策略网络
        for _ in range(self.num_epochs):
            # 计算动作概率
            action_probs = self.actor(states_tensor)
            dist = torch.distributions.Categorical(action_probs)
            
            # 计算新旧策略的比率
            log_probs = dist.log_prob(actions_tensor)
            old_log_probs = dist.log_prob(actions_tensor).detach()
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 计算策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算熵损失
            entropy_loss = -self.entropy_coef * dist.entropy().mean()
            
            # 更新策略网络
            self.actor_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新价值网络
            value_pred = self.critic(states_tensor)
            value_loss = self.value_loss_coef * F.mse_loss(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        # 清空缓冲区
        self.clear_buffer()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def _preprocess_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        预处理状态
        
        Args:
            state: 状态字典
            
        Returns:
            torch.Tensor: 处理后的状态张量
        """
        # 获取当前节点和目标节点
        current_node = state['packet']['current_node']
        destination = state['packet']['destination']
        
        # 获取拓扑信息
        positions = state['topology']['positions']  # [num_satellites, 3]
        velocities = state['topology']['velocities']  # [num_satellites, 3]
        
        # 获取网络信息
        queue_lengths = state['network']['queue_lengths']  # [num_satellites]
        link_states = state['network']['link_states']  # [num_satellites, num_satellites]
        
        # 构造特征向量
        features = []
        
        # 1. 当前节点特征
        features.extend(positions[current_node])  # 位置 (3)
        features.extend(velocities[current_node])  # 速度 (3)
        features.append(queue_lengths[current_node])  # 队列长度 (1)
        
        # 2. 目标节点特征
        features.extend(positions[destination])  # 位置 (3)
        features.extend(velocities[destination])  # 速度 (3)
        features.append(queue_lengths[destination])  # 队列长度 (1)
        
        # 3. 链路状态
        features.extend(link_states[current_node])  # 当前节点的链路状态
        
        # 4. 全局特征
        features.append(float(current_node))  # 当前节点ID
        features.append(float(destination))  # 目标节点ID
        
        # 转换为张量
        state_tensor = torch.FloatTensor(features).to(self.device)
        return state_tensor.unsqueeze(0)  # [1, state_dim]
    
    def clear_buffer(self) -> None:
        """清空经验回放缓冲区"""
        self.buffer = []
    
    def train(self) -> None:
        """设置为训练模式"""
        self.training = True
        self.actor.train()
        self.critic.train()
    
    def eval(self) -> None:
        """设置为评估模式"""
        self.training = False
        self.actor.eval()
        self.critic.eval()
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) 