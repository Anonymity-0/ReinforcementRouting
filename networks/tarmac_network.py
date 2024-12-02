# networks/tarmac_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNet(nn.Module):
    def __init__(self, obs_size, action_dim, hidden_dim, comm_dim, num_heads):
        super(AgentNet, self).__init__()
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.comm_dim = comm_dim
        self.num_heads = num_heads

        self.fc1 = nn.Linear(obs_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, comm_dim)
        self.attention = nn.MultiheadAttention(comm_dim, num_heads)
        self.fc3 = nn.Linear(comm_dim, action_dim)

    def encode(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, obs, messages):
        hidden = self.encode(obs)
        hidden = hidden.unsqueeze(0)  # 添加时间维度
        messages = torch.stack(messages).unsqueeze(1)  # 添加时间维度
        attn_output, _ = self.attention(hidden, messages, messages)
        attn_output = attn_output.squeeze(0)  # 移除时间维度
        action_logits = self.fc3(attn_output)
        return action_logits

class TarMACNetwork(nn.Module):
    def __init__(self, n_agents, obs_sizes, action_dim, hidden_dim, comm_dim, num_heads=2):
        super(TarMACNetwork, self).__init__()
        self.n_agents = n_agents
        self.obs_sizes = obs_sizes
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.comm_dim = comm_dim
        self.num_heads = num_heads
        self.agent_nets = nn.ModuleList()
        for obs_size in obs_sizes:
            self.agent_nets.append(AgentNet(obs_size, action_dim, hidden_dim, comm_dim, num_heads))

    def forward(self, obs_list):
        # 初始化所有智能体的隐藏状态和消息
        hiddens = []
        messages = [torch.zeros(1, self.comm_dim) for _ in range(self.n_agents)]
        actions = []
        # 第一步：每个智能体根据自身观察和消息编码隐藏状态
        for i in range(self.n_agents):
            obs = obs_list[i]
            agent_net = self.agent_nets[i]
            hidden = agent_net.encode(obs)
            hiddens.append(hidden)
        # 第二步：智能体之间进行通信，聚合消息
        for i in range(self.n_agents):
            obs = obs_list[i]
            agent_net = self.agent_nets[i]
            action_logits = agent_net(obs, hiddens)
            actions.append(action_logits)
        return actions
