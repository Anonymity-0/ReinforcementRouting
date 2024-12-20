# agents/mappo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.mappo_network import MAPPOActor, MAPPOCritic
from utils.replay_buffer import ReplayBuffer

class MAPPOAgent:
    def __init__(self, n_agents, obs_sizes, action_dim, hidden_dim, learning_rate, device, multi_agent=True):
        self.n_agents = n_agents
        self.obs_sizes = obs_sizes
        self.action_dim = action_dim
        self.device = device
        self.multi_agent = multi_agent

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
        actions = {}
        if self.multi_agent:
            for i, agent_id in enumerate(observations.keys()):
                obs = observations[agent_id]
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.action_dim)
                else:
                    dist = self.actors[i](obs_tensor)
                    action = dist.sample().item()
                actions[agent_id] = action
        else:
            obs = observations['agent_0']
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            if np.random.rand() < epsilon:
                action = np.random.choice(self.action_dim)
            else:
                dist = self.actors[0](obs_tensor)
                action = dist.sample().item()
            actions['agent_0'] = action
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

        if self.multi_agent:
            # 拼接所有智能体的观察，作为全局状态
            global_states = np.array([np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())]) for obs in batch_obs])
            global_next_states = np.array([np.concatenate([obs[agent_id] for agent_id in sorted(obs.keys())]) for obs in batch_next_obs])
        else:
            # 单智能体
            global_states = np.array([obs['agent_0'] for obs in batch_obs])
            global_next_states = np.array([obs['agent_0'] for obs in batch_next_obs])

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
        for i, actor in enumerate(self.actors):
            if self.multi_agent:
                obs_i = np.array([obs[agent_id] for obs, agent_id in zip(batch_obs, sorted(batch_obs[0].keys()))], dtype=np.float32)
                actions_i = torch.tensor([actions[agent_id] for actions, agent_id in zip(batch_actions, sorted(batch_actions[0].keys()))], dtype=torch.long).to(self.device)
            else:
                obs_i = np.array([obs['agent_0'] for obs in batch_obs], dtype=np.float32)
                actions_i = torch.tensor([actions['agent_0'] for actions in batch_actions], dtype=torch.long).to(self.device)

            obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)
            dist = actor(obs_tensor)
            log_probs = dist.log_prob(actions_i)

            actor_loss = -(log_probs * advantages.detach().squeeze()).mean()
            actor_losses.append(actor_loss)

        critic_loss = advantages.pow(2).mean()

        total_loss = sum(actor_losses) + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
