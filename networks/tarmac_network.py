# networks/tarmac_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            agent_net = self.agent_nets[i]
            # 获取其他智能体的隐藏状态
            other_hiddens = hiddens[:i] + hiddens[i+1:]
            # 智能体 i 接收来自其他智能体的消息
            comm = agent_net.communicate(hiddens[i], other_hiddens)
            # 更新隐藏状态
            hiddens[i] = comm

        # 第三步：每个智能体根据更新后的隐藏状态生成动作
        for i in range(self.n_agents):
            agent_net = self.agent_nets[i]
            action_dist = agent_net.decode(hiddens[i])
            action = action_dist.sample()
            actions.append(action)

        return actions

class AgentNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, comm_dim, num_heads):
        super(AgentNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.comm_dim = comm_dim
        self.num_heads = num_heads

        # 编码器：将观察编码为隐藏状态
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # 消息生成器
        self.message_generator = nn.Linear(hidden_dim, comm_dim * num_heads)

        # 消息选择器
        self.key_generator = nn.Linear(hidden_dim, comm_dim * num_heads)

        # 消息聚合器
        self.attention = nn.MultiheadAttention(embed_dim=comm_dim, num_heads=num_heads, batch_first=True)

        # 解码器：将聚合的隐藏状态解码为动作分布
        self.decoder = nn.Linear(hidden_dim, action_dim)

    def encode(self, obs):
        hidden = F.relu(self.encoder(obs))
        return hidden

    def communicate(self, hidden, other_hiddens):
        # 生成消息和键
        message = self.message_generator(hidden)  # (hidden_dim) -> (comm_dim * num_heads)
        key = self.key_generator(hidden)          # (hidden_dim) -> (comm_dim * num_heads)

        # 处理其他智能体的消息和键
        messages = []
        keys = []
        for other_hidden in other_hiddens:
            other_message = self.message_generator(other_hidden)
            other_key = self.key_generator(other_hidden)
            messages.append(other_message.unsqueeze(0))  # (1, comm_dim * num_heads)
            keys.append(other_key.unsqueeze(0))

        if len(messages) == 0:
            return hidden  # 没有其他智能体，不需要通信

        messages = torch.cat(messages, dim=0)  # (n_agents - 1, comm_dim * num_heads)
        keys = torch.cat(keys, dim=0)          # (n_agents - 1, comm_dim * num_heads)

        # Reshape for MultiheadAttention
        query = message.view(1, 1, -1)  # (batch_size, seq_len, embed_dim)
        keys = keys.view(1, -1, self.comm_dim * self.num_heads)
        messages = messages.view(1, -1, self.comm_dim * self.num_heads)

        # 计算注意力权重并聚合消息
        attn_output, _ = self.attention(query, keys, messages)
        attn_output = attn_output.squeeze(0).squeeze(0)  # (comm_dim * num_heads)

        # 更新隐藏状态
        new_hidden = hidden + attn_output
        return new_hidden

    def decode(self, hidden):
        action_logits = self.decoder(hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        return dist
