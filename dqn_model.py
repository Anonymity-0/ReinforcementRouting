import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 添加缺失的常量定义
INITIAL_EPSILON = 0.9
MIN_EPSILON = 0.1
DECAY_RATE = 0.0001
QUEUE_CAPACITY = 100  # MB

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, leo_names, leo_to_meo):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.leo_names = leo_names
        self.leo_to_meo = leo_to_meo
        
        # 学习参数
        self.gamma = 0.95
        self.epsilon = INITIAL_EPSILON
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = DECAY_RATE
        self.learning_rate = 0.001
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_state(self, env, current_leo, destination_leo):
        """获取当前状态表示"""
        current_metrics = []
        
        # 获取与邻居节点的链路性能指标
        for neighbor in env.leo_nodes[current_leo].connected_satellites:
            metrics = env._calculate_link_metrics(current_leo, neighbor)
            if metrics:
                current_metrics.extend([
                    metrics['delay'] / 100.0,
                    metrics['bandwidth'] / 20.0,
                    metrics['loss'] / 100.0,
                    env.links_dict.get((current_leo, neighbor) if (current_leo, neighbor) in env.links_dict 
                                     else (neighbor, current_leo)).traffic / QUEUE_CAPACITY
                ])
        
        # 添加目标LEO的位置信息和区域信息
        dest_meo = self.leo_to_meo[destination_leo]
        current_meo = self.leo_to_meo[current_leo]
        same_region = 1.0 if dest_meo == current_meo else 0.0
        
        # 获取最小交叉区域大小
        cross_region_size = env.get_cross_region_size(current_leo, destination_leo)
        
        # 组合状态向量
        state = current_metrics + [same_region, cross_region_size]
        
        # 填充固定长度
        max_neighbors = 4  # 假设最多4个邻居
        expected_length = max_neighbors * 4 + 2  # 4个指标 + 区域标识 + 交叉区域大小
        if len(state) < expected_length:
            state.extend([0.0] * (expected_length - len(state)))
        
        # 确保返回的是Tensor
        return torch.FloatTensor(state).to(self.device)

    def get_candidate_actions(self, current_leo, destination, env, available_actions):
        """获取候选动作"""
        # 使用find_k_shortest_paths获取候选路径
        candidate_paths = self._find_k_shortest_paths(current_leo, destination, 3, env.leo_graph)
        candidate_actions = set()
        
        # 从候选路径中提取下一步可能的动作
        for path in candidate_paths:
            if len(path) > 1 and path[0] == current_leo:
                next_leo = path[1]
                action_idx = self.leo_names.index(next_leo)
                if action_idx in available_actions:
                    candidate_actions.add(action_idx)
        
        # 如果没有找到候选动作，返回所有可用动作
        if not candidate_actions:
            return available_actions
        
        return list(candidate_actions)

    def calculate_path_quality(self, metrics):
        """计算路径质量得分"""
        if not metrics:
            return float('-inf')
            
        delay_score = max(0, 1 - metrics['delay'] / 200)  # 延迟越低越好
        bandwidth_score = metrics['bandwidth'] / 20  # 带宽越高越好
        loss_score = max(0, 1 - metrics['loss'] / 100)  # 丢包率越低越好
        
        # 综合得分，权重可调
        return (delay_score * 0.4 + bandwidth_score * 0.4 + loss_score * 0.2)

    def choose_action(self, state, available_actions, env, current_leo, destination, path):
        """选择动作"""
        if len(available_actions) == 0:
            return None
            
        # 移除会导致环路的动作
        non_loop_actions = [a for a in available_actions if self.leo_names[a] not in path]
        if not non_loop_actions:
            return None
            
        # 获取候选动作
        candidate_actions = env.get_candidate_actions(current_leo, destination, non_loop_actions)
        
        # 探索
        if random.random() < self.epsilon:
            if random.random() < 0.8 and candidate_actions:
                return random.choice(candidate_actions)
            return random.choice(non_loop_actions)
        
        # 利用
        state = torch.FloatTensor(state).to(self.device)  # 确保state是Tensor
        with torch.no_grad():
            action_values = self.policy_net(state)
        
        # 只考虑非循环动作中的最大值
        valid_actions = {action: action_values[action].item() for action in non_loop_actions}
        return max(valid_actions.items(), key=lambda x: x[1])[0]

    def memorize(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """经验回放"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # 确保所有状态都是 Tensor
        states = []
        for s in [s[0] for s in batch]:
            if isinstance(s, list):
                states.append(torch.FloatTensor(s).to(self.device))
            else:
                states.append(s)
        states = torch.stack(states)
        
        actions = torch.tensor([s[1] for s in batch], device=self.device)
        rewards = torch.tensor([s[2] for s in batch], device=self.device, dtype=torch.float32)
        
        # 确保所有下一个状态都是 Tensor
        next_states = []
        for s in [s[3] for s in batch]:
            if isinstance(s, list):
                next_states.append(torch.FloatTensor(s).to(self.device))
            else:
                next_states.append(s)
        next_states = torch.stack(next_states)
        
        dones = torch.tensor([s[4] for s in batch], device=self.device, dtype=torch.float32)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 计算损失并更新
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

