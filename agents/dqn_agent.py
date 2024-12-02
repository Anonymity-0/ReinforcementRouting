# agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer

class DQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        q_values = self.fc2(x)
        return q_values

class DQNAgent:
    def __init__(self, obs_size, action_dim, hidden_dim, learning_rate, device):
        self.action_dim = action_dim
        self.device = device
        self.policy_net = DQNetwork(obs_size, action_dim, hidden_dim).to(device)
        self.target_net = DQNetwork(obs_size, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=5000)
    
    def select_action(self, observation, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_dim)
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
            action = torch.argmax(q_values, dim=-1).item()
        return action
    
    def store_experience(self, experience):
        self.replay_buffer.add(experience)
    
    def update(self, batch_size, gamma):
        if len(self.replay_buffer.buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_obs_tensor = torch.tensor(next_observations, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(obs_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_tensor).max(1)[0].unsqueeze(1)
            targets = rewards_tensor + gamma * (1 - dones_tensor) * next_q_values
        
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())