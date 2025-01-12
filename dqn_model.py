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
        # 初始化固定长度的状态向量
        state = np.zeros(18)
        
        try:
            # 1. 获取与邻居节点的链路性能指标（最多4个邻居，每个邻居4个指标）
            neighbor_count = 0
            for neighbor in env.leo_nodes[current_leo].connected_satellites:
                if neighbor_count >= 4:  # 限制最多处理4个邻居
                    break
                    
                metrics = env._calculate_link_metrics(current_leo, neighbor)
                if metrics:
                    # 归一化指标
                    base_idx = neighbor_count * 4
                    state[base_idx] = metrics['delay'] / 100.0  # 延迟归一化
                    state[base_idx + 1] = metrics['bandwidth'] / 20.0  # 带宽归一化
                    state[base_idx + 2] = metrics['loss'] / 100.0  # 丢包率归一化
                    
                    # 获取链路对象并归一化流量
                    link = (env.links_dict.get((current_leo, neighbor)) or 
                           env.links_dict.get((neighbor, current_leo)))
                    if link:
                        state[base_idx + 3] = link.traffic / QUEUE_CAPACITY
                    
                neighbor_count += 1
            
            # 2. 添加目标LEO的位置信息和区域信息
            dest_meo = self.leo_to_meo[destination_leo]
            current_meo = self.leo_to_meo[current_leo]
            state[16] = 1.0 if dest_meo == current_meo else 0.0
            
            # 3. 获取最小交叉区域大小并归一化
            cross_region_size = env.get_cross_region_size(current_leo, destination_leo)
            state[17] = min(cross_region_size / 32.0, 1.0)
            
        except Exception as e:
            print(f"生成状态向量时出错: {str(e)}")
            # 即使出错也确保返回18维向量
        
        # 确保所有值都在[0,1]范围内
        state = np.clip(state, 0, 1)
        
        # 转换为Tensor并确保维度正确
        state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.size(0) != 18:
            # 如果维度不正确，进行填充或截断
            correct_size_tensor = torch.zeros(18, device=self.device)
            correct_size_tensor[:min(18, state_tensor.size(0))] = state_tensor[:min(18, state_tensor.size(0))]
            state_tensor = correct_size_tensor
        
        return state_tensor

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
        # 确保状态是18维的 Tensor
        if isinstance(state, list):
            # 如果状态长度不足18，进行填充
            if len(state) < 18:
                state = state + [0.0] * (18 - len(state))
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor) and state.size(0) < 18:
            # 如果是Tensor但维度不足，进行填充
            padding = torch.zeros(18 - state.size(0), device=self.device)
            state = torch.cat([state, padding])
        
        if isinstance(next_state, list):
            # 如果下一个状态长度不足18，进行填充
            if len(next_state) < 18:
                next_state = next_state + [0.0] * (18 - len(next_state))
            next_state = torch.FloatTensor(next_state).to(self.device)
        elif isinstance(next_state, torch.Tensor) and next_state.size(0) < 18:
            # 如果是Tensor但维度不足，进行填充
            padding = torch.zeros(18 - next_state.size(0), device=self.device)
            next_state = torch.cat([next_state, padding])
        
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """经验回放"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            # 确保状态是18维的 Tensor
            if isinstance(state, list):
                if len(state) < 18:
                    state = state + [0.0] * (18 - len(state))
                state = torch.FloatTensor(state).to(self.device)
            elif isinstance(state, torch.Tensor) and state.size(0) < 18:
                padding = torch.zeros(18 - state.size(0), device=self.device)
                state = torch.cat([state, padding])
            
            if isinstance(next_state, list):
                if len(next_state) < 18:
                    next_state = next_state + [0.0] * (18 - len(next_state))
                next_state = torch.FloatTensor(next_state).to(self.device)
            elif isinstance(next_state, torch.Tensor) and next_state.size(0) < 18:
                padding = torch.zeros(18 - next_state.size(0), device=self.device)
                next_state = torch.cat([next_state, padding])
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # 将列表转换为 Tensor
        states = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

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

