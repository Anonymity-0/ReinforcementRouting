# networks/q_network.py

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        q_values = self.fc2(x)
        return q_values
