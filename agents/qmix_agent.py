# agents/qmix_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from networks.q_network import QNetwork
from networks.mixer_network import MixerNetwork
from utils.replay_buffer import ReplayBuffer
import numpy as np

class QMIXAgent:
    def __init__(self, n_agents, obs_sizes, action_dim, hidden_dim, learning_rate, device):
        self.n_agents = n_agents
        self.obs_sizes = obs_sizes
        self.action_dim = action_dim
        self.device = device

        # 创建 Q 网络和 Mixer 网络，并移动到设备
        self.q_networks = nn.ModuleList([QNetwork(obs_size, action_dim, hidden_dim).to(self.device) for obs_size in obs_sizes])
        self.target_q_networks = nn.ModuleList([QNetwork(obs_size, action_dim, hidden_dim).to(self.device) for obs_size in obs_sizes])
        for target_q_net, q_net in zip(self.target_q_networks, self.q_networks):
            target_q_net.load_state_dict(q_net.state_dict())

        self.mixer_network = MixerNetwork(n_agents, sum(obs_sizes), hidden_dim).to(self.device)
        self.target_mixer_network = MixerNetwork(n_agents, sum(obs_sizes), hidden_dim).to(self.device)
        self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())

        # 定义优化器
        self.params = list(self.q_networks.parameters()) + list(self.mixer_network.parameters())
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=5000)

    def select_action(self, observations, epsilon=0.1):
        actions = []
        for i, obs in enumerate(observations):
            if np.random.rand() < epsilon:
                action = np.random.choice(self.action_dim)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_networks[i](obs_tensor)
                action = torch.argmax(q_values).item()
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

        # 转换为张量并移动到设备
        agent_qs = []
        next_agent_qs = []
        for i in range(self.n_agents):
            obs_i = np.array([obs[i] for obs in batch_obs])
            actions_i = torch.tensor([actions[i] for actions in batch_actions], dtype=torch.long).to(self.device)
            next_obs_i = np.array([obs[i] for obs in batch_next_obs])

            obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)
            next_obs_tensor = torch.tensor(next_obs_i, dtype=torch.float32).to(self.device)

            q_values = self.q_networks[i](obs_tensor)
            q_values = q_values.gather(1, actions_i.unsqueeze(1)).squeeze(1)
            agent_qs.append(q_values)

            with torch.no_grad():
                next_q_values = self.target_q_networks[i](next_obs_tensor)
                max_next_q_values = next_q_values.max(dim=1)[0]
                next_agent_qs.append(max_next_q_values)

        agent_qs = torch.stack(agent_qs, dim=1)
        next_agent_qs = torch.stack(next_agent_qs, dim=1)

        # Mixer 合并 Q 值
        total_q = self.mixer_network(agent_qs)
        with torch.no_grad():
            total_next_q = self.target_mixer_network(next_agent_qs)

        rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch_dones, dtype=torch.float32).to(self.device)
        targets = rewards + gamma * (1 - dones) * total_next_q.squeeze()

        # 计算损失并优化
        loss = nn.MSELoss()(total_q.squeeze(), targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()

        # 更新目标网络
        self.update_target_networks()

    def update_target_networks(self, tau=0.01):
        for target_q_net, q_net in zip(self.target_q_networks, self.q_networks):
            for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_mixer_network.parameters(), self.mixer_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # 保存模型
    def save(self, filepath):
        save_dict = {
            'q_networks': [q_net.state_dict() for q_net in self.q_networks],
            'target_q_networks': [q_net.state_dict() for q_net in self.target_q_networks],
            'mixer_network': self.mixer_network.state_dict(),
            'target_mixer_network': self.target_mixer_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(save_dict, filepath)
        print(f"模型已保存到 {filepath}")

    # 加载模型
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        for q_net, state_dict in zip(self.q_networks, checkpoint['q_networks']):
            q_net.load_state_dict(state_dict)
        for target_q_net, state_dict in zip(self.target_q_networks, checkpoint['target_q_networks']):
            target_q_net.load_state_dict(state_dict)
        self.mixer_network.load_state_dict(checkpoint['mixer_network'])
        self.target_mixer_network.load_state_dict(checkpoint['target_mixer_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"模型已从 {filepath} 加载")
