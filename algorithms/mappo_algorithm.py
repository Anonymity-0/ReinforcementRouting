import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        
        # 计算Actor的状态维度（局部信息,参考PPO）
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
        self.actor_state_dim = 8 + 8 + self.total_satellites + 3
        
        # 计算Critic的状态维度（全局信息）
        # 对每个卫星:
        # 1. 位置 (3)
        # 2. 速度 (3)
        # 3. 队列长度 (1)
        # 4. 轨道面 (1)
        # 5. 链路状态 (total_satellites)
        # 加上任务信息:
        # 6. 当前节点ID (1)
        # 7. 目标节点ID (1)
        self.critic_state_dim = self.total_satellites * (8 + self.total_satellites) + 2
        
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
        
        print(f"Actor状态维度: {self.actor_state_dim}, Critic状态维度: {self.critic_state_dim}, 动作维度: {self.action_dim}")
        
        # 初始化网络
        self.actor = MAPPOActor(self.actor_state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = CentralizedMAPPOCritic(self.critic_state_dim, self.total_satellites, self.hidden_dim).to(self.device)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # 初始化经验回放缓冲区
        self.buffer = []
        
        # 训练模式
        self.training = True

        # 定义轨道面分组（从PPO复制）
        self.ORBITAL_PLANES = {
            1: [145, 143, 140, 148, 150, 153, 144, 149, 146, 142, 157],
            2: [134, 141, 137, 116, 135, 151, 120, 113, 138, 130, 131],
            3: [117, 168, 180, 123, 126, 167, 171, 121, 118, 172, 173],
            4: [119, 122, 128, 107, 132, 129, 100, 133, 125, 136, 139],
            5: [158, 160, 159, 163, 165, 166, 154, 164, 108, 155, 156],
            6: [102, 112, 104, 114, 103, 109, 106, 152, 147, 110, 111]
        }
    
    def get_next_hop(self, current_node: int, destination: int, state: Dict) -> int:
        """
        获取下一跳节点（分散执行）
        
        Args:
            current_node: 当前节点
            destination: 目标节点
            state: 状态字典
            
        Returns:
            int: 下一跳节点ID
        """
        # 如果已到达目标节点
        if current_node == destination:
            return current_node
            
        # 预处理状态（只使用局部信息）
        state_tensor = self._preprocess_state(state, for_critic=False)
        
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
        更新策略和价值网络（中心化训练）
        
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
        """更新网络（中心化训练）"""
        # 从缓冲区中采样数据
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        
        # 预处理状态（Actor使用局部信息，Critic使用全局信息）
        actor_states = torch.cat([self._preprocess_state(s, for_critic=False) for s in states])
        critic_states = torch.cat([self._preprocess_state(s, for_critic=True) for s in states])
        next_critic_states = torch.cat([self._preprocess_state(s, for_critic=True) for s in next_states])
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # [batch_size, 1]
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [batch_size, 1]
        
        # 计算优势函数
        with torch.no_grad():
            values = self.critic(critic_states)  # [batch_size, 1]
            next_values = self.critic(next_critic_states)  # [batch_size, 1]
            
            # 计算TD误差
            deltas = rewards_tensor + self.gamma * next_values * (1 - dones_tensor) - values
            
            # 计算GAE优势
            advantages = torch.zeros_like(deltas)  # [batch_size, 1]
            advantage = 0.0
            for t in reversed(range(len(self.buffer))):
                advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones_tensor[t]) * advantage
                advantages[t] = advantage
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算目标值
            returns = advantages + values  # [batch_size, 1]
        
        # 更新策略网络
        for _ in range(self.num_epochs):
            # 计算动作概率
            action_probs = self.actor(actor_states)
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
            value_pred = self.critic(critic_states)
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
    
    def _preprocess_state(self, state: Dict[str, Any], for_critic: bool = False) -> torch.Tensor:
        """
        预处理状态，遵循CTDE原则
        
        Args:
            state: 状态字典
            for_critic: 是否为critic准备状态（训练时使用全局信息）
            
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
        link_states = state['network']['link_states']  # [num_satellites]
        
        # 获取当前节点和目标节点的轨道面
        current_plane = None
        destination_plane = None
        for plane_id, satellites in self.ORBITAL_PLANES.items():
            if current_node in satellites:
                current_plane = plane_id
            if destination in satellites:
                destination_plane = plane_id
        
        if for_critic:
            # Critic使用全局状态（中心化训练）
            features = []
            # 对每个卫星
            for i in range(self.total_satellites):
                # 1. 位置 (3)
                features.extend(positions[i])
                # 2. 速度 (3)
                features.extend(velocities[i])
                # 3. 队列长度 (1)
                features.append(queue_lengths[i])
                # 4. 轨道面 (1)
                plane = 0
                for plane_id, satellites in self.ORBITAL_PLANES.items():
                    if i in satellites:
                        plane = plane_id
                        break
                features.append(float(plane))
                # 5. 链路状态 (total_satellites)
                # 获取与其他卫星的链路状态
                topology = state['topology']['object']
                link_qualities = []
                for j in range(self.total_satellites):
                    if i != j and topology._can_establish_link(i, j):
                        link_qualities.append(topology.get_link_quality(i, j))
                    else:
                        link_qualities.append(0.0)
                features.extend(link_qualities)
            
            # 6. 当前节点ID (1)
            features.append(float(current_node))
            # 7. 目标节点ID (1)
            features.append(float(destination))
            
            expected_dim = self.total_satellites * (8 + self.total_satellites) + 2
        else:
            # Actor只使用局部信息（分散执行）
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
            
            # 3. 链路状态
            # 获取当前节点与其他卫星的链路状态
            topology = state['topology']['object']
            link_qualities = []
            for i in range(self.total_satellites):
                if current_node != i and topology._can_establish_link(current_node, i):
                    link_qualities.append(topology.get_link_quality(current_node, i))
                else:
                    link_qualities.append(0.0)
            features.extend(link_qualities)  # 链路状态 (total_satellites)
            
            # 4. 全局特征
            features.append(float(current_node))  # 当前节点ID (1)
            features.append(float(destination))  # 目标节点ID (1)
            features.append(float(abs(current_plane - destination_plane) if current_plane is not None and destination_plane is not None else 0))  # 轨道面距离 (1)
            
            expected_dim = 8 + 8 + self.total_satellites + 3
        
        # 转换为张量
        state_tensor = torch.FloatTensor(features).to(self.device)
        
        # 检查维度
        assert len(features) == expected_dim, f"状态维度不匹配: 期望 {expected_dim}, 实际 {len(features)}"
        
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