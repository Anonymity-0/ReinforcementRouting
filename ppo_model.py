import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from config import *

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor网络(策略网络)
        self.actor = nn.Sequential(
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络(价值网络)
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOAgent:
    def __init__(self, state_size, action_size, leo_names, leo_to_meo):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
        
        self.leo_names = leo_names
        self.leo_to_meo = leo_to_meo
        
        # PPO超参数
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # 价值损失系数
        self.c2 = 0.01  # 熵损失系数
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
    def choose_action(self, state, available_actions, env, current_leo, destination, path):
        """选择动作"""
        if not available_actions:
            return None
        
        try:
            # 移除会导致环路的动作
            non_loop_actions = [a for a in available_actions 
                               if a < len(self.leo_names) and  # 确保动作索引有效
                               self.leo_names[a] not in path]
            if not non_loop_actions:
                return None
            
            # 获取候选动作
            candidate_actions = env.get_candidate_actions(current_leo, destination, non_loop_actions)
            
            # 将状态转换为tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取动作概率分布
            with torch.no_grad():
                action_probs, _ = self.actor_critic(state)
                action_probs = action_probs.squeeze()
            
            # 只考虑有效动作的概率
            valid_probs = torch.zeros_like(action_probs)
            valid_actions = candidate_actions if candidate_actions else non_loop_actions
            for action in valid_actions:
                if action < len(action_probs):
                    valid_probs[action] = action_probs[action]
            
            # 如果没有有效概率，返回None
            if valid_probs.sum() == 0:
                return None
            
            # 归一化概率
            valid_probs = valid_probs / valid_probs.sum()
            
            # 采样动作
            action = torch.multinomial(valid_probs, 1).item()
            
            # 检查动作有效性
            if action >= len(self.leo_names):
                print(f"无效的动作索引: {action}, 最大允许值: {len(self.leo_names)-1}")
                return None
            
            return action
            
        except Exception as e:
            print(f"选择动作时出错: {str(e)}")
            return None
        
    def update(self, next_value):
        """更新策略"""
        returns = []
        advantages = []
        R = next_value
        
        # 计算优势和回报
        for r, v, mask in zip(reversed(self.rewards), reversed(self.values), reversed(self.masks)):
            R = r + GAMMA * R * mask
            advantage = R - v
            returns.append(R)
            advantages.append(advantage)
            
        returns = torch.tensor(list(reversed(returns))).to(self.device)
        advantages = torch.tensor(list(reversed(advantages))).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        old_states = torch.stack(self.states)
        old_actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs).detach()
        
        for _ in range(10):  # PPO epochs
            action_probs, values = self.actor_critic(old_states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算surrogate损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = self.c1 * (returns - values).pow(2).mean()
            
            # 熵损失(用于鼓励探索)
            entropy_loss = -self.c2 * dist.entropy().mean()
            
            # 总损失
            total_loss = actor_loss + value_loss + entropy_loss
            
            # 更新网络
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = [] 