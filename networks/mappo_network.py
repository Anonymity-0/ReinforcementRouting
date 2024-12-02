# networks/mappo_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPPOActor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(MAPPOActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        return dist

class MAPPOCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MAPPOCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state):
        x = F.relu(self.fc1(global_state))
        value = self.value_head(x)
        return value
