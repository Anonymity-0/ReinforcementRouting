# agents/tarmac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.tarmac_network import TarMACNetwork
from utils.replay_buffer import ReplayBuffer

class TarMACAgent:
    def __init__(self, n_agents, obs_sizes, action_dim, hidden_dim, comm_dim, num_heads, learning_rate, device):
        self.n_agents = n_agents
        self.obs_sizes = obs_sizes
        self.action_dim = action_dim
        self.device = device

        # 创建 TarMAC 网络
        self.network = TarMACNetwork(n_agents, obs_sizes, action_dim, hidden_dim, comm_dim, num_heads).to(device)

        # 定义优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size=5000)

    def select_action(self, observations):
        obs_tensors = [torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) for obs in observations]
        self.network.eval()
        with torch.no_grad():
            actions = self.network(obs_tensors)
        self.network.train()
        return [action.item() for action in actions]

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

        # 转换为张量
        total_loss = 0
        for i in range(self.n_agents):
            obs_i = np.array([obs[i] for obs in batch_obs])
            actions_i = torch.tensor([actions[i] for actions in batch_actions], dtype=torch.long).to(self.device)
            rewards_i = torch.tensor(batch_rewards, dtype=torch.float32).to(self.device)
            dones_i = torch.tensor(batch_dones, dtype=torch.float32).to(self.device)

            obs_tensor = torch.tensor(obs_i, dtype=torch.float32).to(self.device)

            # 计算当前策略的动作分布
            dist = self.network.agent_nets[i].decode(self.network.agent_nets[i].encode(obs_tensor))
            log_probs = dist.log_prob(actions_i)

            # 假设使用蒙特卡洛方法估计回报（可以根据需要使用优势函数）
            returns = []
            G = 0
            for r, done in zip(reversed(rewards_i), reversed(dones_i)):
                if done:
                    G = 0
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            # 计算损失
            loss = -torch.mean(log_probs * returns)
            total_loss += loss

        # 优化网络
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
