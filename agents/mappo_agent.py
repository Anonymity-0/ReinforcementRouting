# agents/mappo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.mappo_network import MAPPOActor, MAPPOCritic
from utils.replay_buffer import ReplayBuffer

class MAPPOAgent:
    def __init__(self, n_agents, obs_sizes, action_dim, hidden_dim, learning_rate, device):
        self.n_agents = n_agents
        self.obs_sizes = obs_sizes  # 每个智能体的观测维度
        self.action_dim = action_dim
        self.device = device

        # 创建 Actor 和 Critic 网络
        self.actors = nn.ModuleList([MAPPOActor(obs_size, action_dim, hidden_dim).to(device) for obs_size in obs_sizes])
        self.critic = MAPPOCritic(sum(obs_sizes), hidden_dim).to(device)

        # 定义优化器
        self.actor_params = []
        for actor in self.actors:
            self.actor_params += list(actor.parameters())
        self.optimizer = optim.Adam(self.actor_params + list(self.critic.parameters()), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=5000)

    def select_action(self, observations, epsilon=0.0):
        actions = []
        for i, obs in enumerate(observations):
            if np.random.rand() < epsilon:
                action = np.random.choice(self.action_dim)
                actions.append(action)
            else:
                # 从观测字典中提取数值特征
                cache_states = obs.get('cache_states', [])
                load_balance = obs.get('load_balance', 0.0)
                connectivity = obs.get('connectivity', 0.0)
                avg_cache = obs.get('avg_cache', 0.0)
                src_in_region = float(obs.get('src_in_region', False))
                dst_in_region = float(obs.get('dst_in_region', False))

                # 合并所有特征为一个数组
                combined_obs = np.concatenate([
                    np.array(cache_states, dtype=np.float32),
                    np.array([load_balance, connectivity, avg_cache, src_in_region, dst_in_region], dtype=np.float32)
                ])

                # 将合并后的数组转换为张量
                obs_tensor = torch.tensor(combined_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                dist = self.actors[i](obs_tensor)
                action = dist.sample().item()
                actions.append(action)
        return actions

    def store_experience(self, experience):
        self.replay_buffer.add(experience)

    def update(self, batch_size, gamma):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        batch_obs = [item[0] for item in batch]
        batch_actions = [item[1] for item in batch]
        batch_rewards = [item[2] for item in batch]
        batch_next_obs = [item[3] for item in batch]
        batch_dones = [item[4] for item in batch]

        # 处理观测值，提取特征并拼接
        global_states = [np.concatenate([self._process_observation(obs[i]) for i in range(self.n_agents)]) for obs in batch_obs]
        global_next_states = [np.concatenate([self._process_observation(obs[i]) for i in range(self.n_agents)]) for obs in batch_next_obs]

        # 转换为张量
        global_states = torch.tensor(global_states, dtype=torch.float32).to(self.device)
        global_next_states = torch.tensor(global_next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 计算价值函数
        values = self.critic(global_states)
        next_values = self.critic(global_next_states).detach()

        # 计算优势函数
        targets = rewards + gamma * (1 - dones) * next_values
        advantages = targets - values

        # 更新 Actor 和 Critic
        actor_losses = []
        for i in range(self.n_agents):
            obs_i = [self._process_observation(obs[i]) for obs in batch_obs]
            obs_i = np.array(obs_i, dtype=np.float32)
            actions_i = torch.tensor([actions[i] for actions in batch_actions], dtype=torch.long).to(self.device)

            obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)
            dist = self.actors[i](obs_tensor)
            log_probs = dist.log_prob(actions_i)

            actor_loss = -(log_probs * advantages.detach().squeeze()).mean()
            actor_losses.append(actor_loss)

        critic_loss = advantages.pow(2).mean()

        total_loss = sum(actor_losses) + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def _process_observation(self, obs):
        # 处理单个智能体的观测，转换为一维numpy数组
        cache_states = obs.get('cache_states', [])
        load_balance = obs.get('load_balance', 0.0)
        connectivity = obs.get('connectivity', 0.0)
        avg_cache = obs.get('avg_cache', 0.0)
        src_in_region = float(obs.get('src_in_region', False))
        dst_in_region = float(obs.get('dst_in_region', False))

        # 合并所有特征为一个数组
        combined_obs = np.concatenate([
            np.array(cache_states, dtype=np.float32),
            np.array([load_balance, connectivity, avg_cache, src_in_region, dst_in_region], dtype=np.float32)
        ])
        return combined_obs

    # 保存模型
    def save(self, filepath):
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"模型已保存到 {filepath}")

    # 加载模型
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"模型已从 {filepath} 加载")
