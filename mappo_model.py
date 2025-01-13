import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from config import *

class MAPPONetwork(nn.Module):
    def __init__(self, state_size, action_size, n_agents):
        super(MAPPONetwork, self).__init__()
        self.n_agents = n_agents
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size * n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor网络(每个智能体一个)
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, action_size),
                nn.Softmax(dim=-1)
            ) for _ in range(n_agents)
        ])
        
        # Critic网络(共享)
        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )
        
    def forward(self, states):
        batch_size = states.size(0)
        states = states.view(batch_size, -1)  # 展平所有智能体的状态
        features = self.feature_layer(states)
        
        # 为每个智能体生成动作概率
        action_probs = [actor(features) for actor in self.actors]
        value = self.critic(features)
        
        return action_probs, value

class MAPPOAgent:
    def __init__(self, state_size, action_size, n_agents, leo_names, leo_to_meo):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = MAPPONetwork(state_size, action_size, n_agents).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        
        self.n_agents = n_agents
        self.leo_names = leo_names
        self.leo_to_meo = leo_to_meo
        
        # MAPPO超参数
        self.clip_epsilon = 0.2
        self.c1 = 1.0
        self.c2 = 0.01
        
        # 经验缓冲区(每个智能体)
        self.states = [[] for _ in range(n_agents)]
        self.actions = [[] for _ in range(n_agents)]
        self.rewards = [[] for _ in range(n_agents)]
        self.values = [[] for _ in range(n_agents)]
        self.log_probs = [[] for _ in range(n_agents)]
        self.masks = [[] for _ in range(n_agents)]
        
    def get_states(self, env, current_leos, destination_leos):
        """获取所有智能体的状态"""
        states = []
        for current, dest in zip(current_leos, destination_leos):
            state = env.get_state(current, dest)
            states.append(state)
        return torch.FloatTensor(states).to(self.device)
        
    def choose_actions(self, states, available_actions_list, env, current_leos, destinations, paths):
        """为所有智能体选择动作"""
        with torch.no_grad():
            action_probs_list, values = self.network(states)
            
        actions = []
        for i in range(self.n_agents):
            available_actions = available_actions_list[i]
            path = paths[i]
            
            # 移除导致环路的动作
            non_loop_actions = [a for a in available_actions if self.leo_names[a] not in path]
            if not non_loop_actions:
                actions.append(None)
                continue
                
            # 获取候选动作
            candidate_actions = env.get_candidate_actions(
                current_leos[i], 
                destinations[i],
                non_loop_actions
            )
            
            # 只考虑可用动作的概率分布
            action_probs = action_probs_list[i]
            mask = torch.zeros_like(action_probs)
            mask[candidate_actions] = 1
            masked_probs = action_probs * mask
            masked_probs = masked_probs / masked_probs.sum()
            
            dist = Categorical(masked_probs)
            action = dist.sample()
            
            # 存储经验
            self.states[i].append(states[i])
            self.actions[i].append(action)
            self.values[i].append(values[i])
            self.log_probs[i].append(dist.log_prob(action))
            
            actions.append(action.item())
            
        return actions
        
    def update(self, next_values):
        """更新所有智能体的策略"""
        for i in range(self.n_agents):
            returns = []
            advantages = []
            R = next_values[i]
            
            # 计算每个智能体的优势和回报
            for r, v, mask in zip(reversed(self.rewards[i]), 
                                reversed(self.values[i]),
                                reversed(self.masks[i])):
                R = r + GAMMA * R * mask
                advantage = R - v
                returns.append(R)
                advantages.append(advantage)
                
            returns = torch.tensor(list(reversed(returns))).to(self.device)
            advantages = torch.tensor(list(reversed(advantages))).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 准备数据
            old_states = torch.stack(self.states[i])
            old_actions = torch.stack(self.actions[i])
            old_log_probs = torch.stack(self.log_probs[i]).detach()
            
            # MAPPO更新
            for _ in range(10):
                action_probs_list, values = self.network(old_states)
                action_probs = action_probs_list[i]
                
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(old_actions)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.c1 * (returns - values).pow(2).mean()
                entropy_loss = -self.c2 * dist.entropy().mean()
                
                total_loss = actor_loss + value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # 清空所有智能体的缓冲区
        for i in range(self.n_agents):
            self.states[i] = []
            self.actions[i] = []
            self.rewards[i] = []
            self.values[i] = []
            self.log_probs[i] = []
            self.masks[i] = [] 