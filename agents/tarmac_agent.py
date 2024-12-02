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
