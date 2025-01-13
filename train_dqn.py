from train_base import BaseTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_freq = 1000
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=100000)
        
        # 创建在线网络和目标网络
        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps_done = 0
        
    def choose_action(self, state, available_actions, env=None, current_leo=None, destination=None, path=None):
        """选择动作
        
        Args:
            state: 当前状态
            available_actions: 可用动作列表
            env: 环境实例（可选）
            current_leo: 当前LEO卫星（可选）
            destination: 目标LEO卫星（可选）
            path: 当前路径（可选）
            
        Returns:
            int: 选择的动作索引
        """
        if random.random() < self.epsilon:
            # 随机探索
            return random.choice(available_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            
            # 只考虑可用动作中的最大Q值
            masked_q_values = q_values.clone()
            mask = torch.ones_like(masked_q_values) * float('-inf')
            mask[0, available_actions] = 0
            masked_q_values += mask
            
            return masked_q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return
        
        # 从经验回放中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class DQNTrainer(BaseTrainer):
    def _run_episode(self, source, destination):
        path = [source]
        total_reward = 0
        current_leo = source
        
        # 记录性能指标
        metrics = {
            'delays': [],
            'bandwidths': [],
            'loss_rates': [],
            'queue_utilizations': []
        }
        
        while len(path) < MAX_PATH_LENGTH:
            state = self.agent.get_state(self.env, current_leo, destination)
            available_actions = self.env.get_available_actions(current_leo)
            
            if not available_actions:
                break
                
            action = self.agent.choose_action(state, available_actions, 
                                            self.env, current_leo, destination, path)
            if action is None:
                break
                
            next_state, reward, done, info = self.env.step(current_leo, action, path)
            
            # 记录性能指标
            link_stats = info.get('link_stats', {})
            metrics['delays'].append(link_stats.get('delay', 0))
            metrics['bandwidths'].append(link_stats.get('bandwidth', 0))
            metrics['loss_rates'].append(link_stats.get('loss', 0))
            metrics['queue_utilizations'].append(link_stats.get('queue_utilization', 0))
            
            # 存储经验
            self.agent.memorize(state, action, reward, next_state, done)
            
            # 更新状态和奖励
            total_reward += reward
            current_leo = list(self.env.leo_nodes.keys())[action]
            path.append(current_leo)
            
            # 经验回放
            self.agent.replay(BATCH_SIZE)
            
            if done or current_leo == destination:
                break
        
        return {
            'path': path,
            'total_reward': total_reward,
            'metrics': metrics
        }
    
    def _save_checkpoint(self, episode, rewards, stats):
        torch.save({
            'episode': episode,
            'model_state_dict': self.agent.policy_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon,
            'rewards': rewards,
            'performance_stats': dict(stats)
        }, f'models/dqn_checkpoint_episode_{episode+1}.pth')
    
    def _save_final_model(self):
        torch.save(self.agent.policy_net.state_dict(), 'models/dqn_final_model.pth') 