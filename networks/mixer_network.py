# networks/mixer_network.py

import torch
import torch.nn as nn

class MixerNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim):
        super(MixerNetwork, self).__init__()
        self.n_agents = n_agents
        self.fc1 = nn.Linear(n_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, q_values):
        x = torch.relu(self.fc1(q_values))
        total_q = self.fc2(x)
        return total_q
