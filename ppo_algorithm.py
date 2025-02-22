import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from .routing_interface import RoutingAlgorithm
from .networks.ppo_nets import PPOActor, PPOCritic
import torch.nn.functional as F
import random

class PPOAlgorithm(RoutingAlgorithm):
    """PPO算法实现"""
    
    # 定义轨道面分组
    ORBITAL_PLANES = {
        1: [145, 143, 140, 148, 150, 153, 144, 149, 146, 142, 157],
        2: [134, 141, 137, 116, 135, 151, 120, 113, 138, 130, 131],
        3: [117, 168, 180, 123, 126, 167, 171, 121, 118, 172, 173],
        4: [119, 122, 128, 107, 132, 129, 100, 133, 125, 136, 139],
        5: [158, 160, 159, 163, 165, 166, 154, 164, 108, 155, 156],
        6: [102, 112, 104, 114, 103, 109, 106, 152, 147, 110, 111]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化PPO算法
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # 获取网络参数
        self.total_satellites = config['total_satellites']
        self.max_buffer_size = config['max_buffer_size']
        self.max_queue_length = config['max_queue_length']
        
        # 获取PPO参数
        self.learning_rate = config.get('learning_rate', 3.0e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_param = config.get('clip_param', 0.2)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.buffer_size = config.get('buffer_size', 2048)
        self.epsilon = config.get('epsilon', 0.3)  # 增大初始探索率
        
        # 获取网络参数
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # 计算状态维度
        # 状态包括:
        # 1. 当前节点特征 (3 + 3 + 1 + 1 = 8)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        #    - 轨道面 (1)
        # 2. 目标节点特征 (3 + 3 + 1 + 1 = 8)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        #    - 轨道面 (1)
        # 3. 链路状态 (total_satellites)
        # 4. 全局特征 (3)
        #    - 当前节点ID (1)
        #    - 目标节点ID (1)
        #    - 轨道面距离 (1)
        self.state_dim = 8 + 8 + self.total_satellites + 3
        
        # 动作维度为卫星数量
        self.action_dim = self.total_satellites
        
        # 初始化策略网络和价值网络
        self.policy_net = PPOActor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.value_net = PPOCritic(self.state_dim, self.hidden_dim).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': self.learning_rate},
            {'params': self.value_net.parameters(), 'lr': self.learning_rate * 3}
        ])
        
        # 初始化经验回放缓冲区
        self.buffer = []
        
        # 训练模式
        self.training = True
    
    def get_next_hop(self, current_node: int, destination: int, state: Dict) -> int:
        """
        获取下一跳节点
        
        Args:
            current_node: 当前节点
            destination: 目标节点
            state: 状态字典
            
        Returns:
            int: 下一跳节点ID
        """
        # 如果已到达目标节点，返回当前节点
        if current_node == destination:
            return current_node
        
        # 获取有效动作（使用topology的get_valid_neighbors方法）
        valid_actions = state['topology']['object'].get_valid_neighbors(current_node)
            
        # 如果没有有效动作，返回当前节点
        if not valid_actions:
            return current_node
        
        # 预处理状态数据
        state_tensor = self._preprocess_state(state)
        
        # 探索模式：以epsilon的概率随机选择一个有效动作
        if self.training and random.random() < self.epsilon:
            next_hop = random.choice(valid_actions)
            return next_hop
        
        # 利用模式：选择概率最高的有效动作
        with torch.no_grad():
            # 获取所有动作的logits
            all_logits = self.policy_net(state_tensor)
            
            # 只提取有效动作的logits
            valid_indices = torch.tensor(valid_actions, device=state_tensor.device)
            valid_logits = all_logits[..., valid_indices]
            
            # 对有效动作的logits进行softmax
            valid_probs = F.softmax(valid_logits, dim=-1)
            
            # 选择概率最高的动作
            action_idx = torch.argmax(valid_probs).item()
            next_hop = valid_actions[action_idx]
        
        return next_hop
    
    def update(self, states: List[Dict], actions: np.ndarray, rewards: np.ndarray,
             next_states: List[Dict], dones: np.ndarray) -> None:
        """
        更新算法
        
        Args:
            states: 状态列表
            actions: 动作数组
            rewards: 奖励数组
            next_states: 下一个状态列表
            dones: 终止标志数组
        """
        # 将数据添加到缓冲区
        self.buffer.extend(zip(states, actions, rewards, next_states, dones))
        
        # 如果缓冲区达到更新条件
        if len(self.buffer) >= self.buffer_size:
            self._update_networks()
            self.clear_buffer()
    
    def _update_networks(self) -> None:
        """更新策略网络和价值网络"""
        # 预处理数据
        states = [s for s, _, _, _, _ in self.buffer]
        actions = torch.FloatTensor([a for _, a, _, _, _ in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in self.buffer]).to(self.device)
        next_states = [s for _, _, _, s, _ in self.buffer]
        dones = torch.FloatTensor([d for _, _, _, _, d in self.buffer]).to(self.device)
        
        # 批量处理状态
        states_tensor = torch.cat([self._preprocess_state(s) for s in states])
        next_states_tensor = torch.cat([self._preprocess_state(s) for s in next_states])
        
        # 计算优势函数
        with torch.no_grad():
            values = self.value_net(states_tensor)
            next_values = self.value_net(next_states_tensor)
            
            # 计算TD误差和GAE优势
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(deltas)
            advantage = 0.0
            for t in reversed(range(len(self.buffer))):
                advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
                advantages[t] = advantage
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values
        
        # 获取每个状态的有效动作
        valid_actions_list = []
        for state in states:
            current_node = state['packet']['current_node']
            valid_actions = state['topology']['object'].get_valid_neighbors(current_node)
            valid_actions_list.append(valid_actions if valid_actions else [-1])
        
        # 计算旧的动作概率
        with torch.no_grad():
            old_probs = []
            for i, state_tensor in enumerate(states_tensor):
                logits = self.policy_net(state_tensor.unsqueeze(0))
                mask = torch.full_like(logits, float('-inf'))
                valid_indices = torch.tensor(valid_actions_list[i], device=self.device)
                mask[..., valid_indices] = 0
                masked_logits = logits + mask
                probs = F.softmax(masked_logits, dim=-1)
                old_probs.append(probs)
            old_probs = torch.cat(old_probs, dim=0)
        
        # 使用小批量更新
        indices = np.random.permutation(len(self.buffer))
        for start_idx in range(0, len(self.buffer), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            # 获取批次数据
            batch_states = states_tensor[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_old_probs = old_probs[batch_indices]
            batch_valid_actions = [valid_actions_list[i] for i in batch_indices]
            
            # 计算新的动作概率
            logits = self.policy_net(batch_states)
            new_probs = []
            for i, valid_actions in enumerate(batch_valid_actions):
                mask = torch.full_like(logits[i:i+1], float('-inf'))
                valid_indices = torch.tensor(valid_actions, device=self.device)
                mask[..., valid_indices] = 0
                masked_logits = logits[i:i+1] + mask
                probs = F.softmax(masked_logits, dim=-1)
                new_probs.append(probs)
            new_probs = torch.cat(new_probs, dim=0)
            
            # 计算损失
            action_indices = batch_actions.long().unsqueeze(1)
            old_action_probs = torch.gather(batch_old_probs, 1, action_indices).squeeze()
            new_action_probs = torch.gather(new_probs, 1, action_indices).squeeze()
            ratio = new_action_probs / (old_action_probs + 1e-8)
            
            # 计算策略损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算熵损失
            entropy_loss = -self.entropy_coef * (-(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1)).mean()
            
            # 计算价值损失
            value_pred = self.value_net(batch_states)
            value_loss = self.value_loss_coef * F.mse_loss(value_pred, batch_returns)
            
            # 计算总损失并更新
            total_loss = policy_loss + value_loss + entropy_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def _preprocess_state(self, state: Dict) -> torch.Tensor:
        """
        预处理状态数据
        
        Args:
            state: 状态字典
            
        Returns:
            torch.Tensor: 预处理后的状态张量
        """
        # 获取当前节点和目标节点
        current_node = state['packet']['current_node']
        destination = state['packet']['destination']
        
        # 获取拓扑信息
        positions = state['topology']['positions']  # [num_satellites, 3]
        velocities = state['topology']['velocities']  # [num_satellites, 3]
        
        # 获取网络信息
        queue_lengths = state['network']['queue_lengths']  # [num_satellites]
        link_states = state['network']['link_states']  # [num_satellites]
        
        # 获取当前节点和目标节点的轨道面
        current_plane = None
        destination_plane = None
        for plane_id, satellites in self.ORBITAL_PLANES.items():
            if current_node in satellites:
                current_plane = plane_id
            if destination in satellites:
                destination_plane = plane_id
        
        # 构造特征向量
        features = []
        
        # 1. 当前节点特征
        features.extend(positions[current_node])  # 位置 (3)
        features.extend(velocities[current_node])  # 速度 (3)
        features.append(queue_lengths[current_node])  # 队列长度 (1)
        features.append(float(current_plane if current_plane is not None else 0))  # 轨道面 (1)
        
        # 2. 目标节点特征
        features.extend(positions[destination])  # 位置 (3)
        features.extend(velocities[destination])  # 速度 (3)
        features.append(queue_lengths[destination])  # 队列长度 (1)
        features.append(float(destination_plane if destination_plane is not None else 0))  # 轨道面 (1)
        
        # 3. 链路状态特征
        features.extend(link_states)  # 链路状态 (num_satellites)
        
        # 4. 全局特征
        features.append(float(current_node))  # 当前节点ID (1)
        features.append(float(destination))  # 目标节点ID (1)
        features.append(float(abs(current_plane - destination_plane) if current_plane is not None and destination_plane is not None else 0))  # 轨道面距离 (1)
        
        # 转换为张量并确保类型正确
        state_tensor = torch.FloatTensor(features).to(self.device)
        
        # 检查维度
        expected_dim = 8 + 8 + self.total_satellites + 3
        assert state_tensor.shape[0] == expected_dim, f"状态维度不匹配: 期望 {expected_dim}, 实际 {state_tensor.shape[0]}"
        
        return state_tensor.unsqueeze(0)  # [1, state_dim]
    
    def clear_buffer(self) -> None:
        """清空经验回放缓冲区"""
        self.buffer = []
    
    def train(self) -> None:
        """设置为训练模式"""
        self.training = True
        self.policy_net.train()
        self.value_net.train()
    
    def eval(self) -> None:
        """设置为评估模式"""
        self.training = False
        self.policy_net.eval()
        self.value_net.eval()
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 模型保存路径
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型加载路径
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])