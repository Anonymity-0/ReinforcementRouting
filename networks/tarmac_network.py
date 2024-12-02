# networks/tarmac_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.attention = nn.MultiheadAttention(embed_dim=comm_dim, num_heads=num_heads)

        self.fc3 = nn.Linear(comm_dim, action_dim)

    def encode(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return x  # 输出形状：[batch_size, comm_dim]

    def forward(self, obs, messages):
        # obs: [batch_size, obs_size]
        # messages: [n_agents - 1, batch_size, comm_dim]

        hidden = self.encode(obs)  # [batch_size, comm_dim]
        hidden = hidden.unsqueeze(0)  # [1, batch_size, comm_dim]，作为 query

        # messages 已经是 [n_agents - 1, batch_size, comm_dim]，可以直接作为 key 和 value

        # 计算注意力
        attn_output, _ = self.attention(hidden, messages, messages)
        attn_output = attn_output.squeeze(0)  # [batch_size, comm_dim]

        action_logits = self.fc3(attn_output)  # [batch_size, action_dim]

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
        batch_size = obs_list[0].shape[0]  # 假设所有观察具有相同的 batch_size

        # 第一步：每个智能体根据自身观察编码隐藏状态
        hiddens = []
        for i in range(self.n_agents):
            obs = obs_list[i]  # [batch_size, obs_size]
            agent_net = self.agent_nets[i]
            hidden = agent_net.encode(obs)  # [batch_size, comm_dim]
            hiddens.append(hidden)

        # 将所有隐藏状态堆叠，形成消息集合
        # messages: [n_agents, batch_size, comm_dim]
        messages = torch.stack(hiddens, dim=0)

        actions = []
        for i in range(self.n_agents):
            obs = obs_list[i]  # [batch_size, obs_size]
            agent_net = self.agent_nets[i]

            # 排除自身的消息
            other_messages = torch.cat([messages[:i], messages[i+1:]], dim=0)  # [n_agents - 1, batch_size, comm_dim]

            # 计算动作 logits
            action_logits = agent_net(obs, other_messages)  # [batch_size, action_dim]
            actions.append(action_logits)

        return actions  # 长度为 n_agents 的列表，每个元素形状为 [batch_size, action_dim]

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
        # observations 是一个列表，长度为 n_agents，每个元素是对应智能体的观察
        # 将每个观察转换为张量，并添加批量维度
        obs_tensors = [torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device) for obs in observations]
        # obs_tensors 中的每个元素形状为 [1, obs_size]
        self.network.eval()
        with torch.no_grad():
            # action_logits_list 是一个长度为 n_agents 的列表，每个元素形状为 [1, action_dim]
            action_logits_list = self.network(obs_tensors)
        self.network.train()
        actions = []
        for action_logits in action_logits_list:
            # action_logits: [1, action_dim]
            action_probs = torch.softmax(action_logits, dim=-1)  # [1, action_dim]
            action = torch.argmax(action_probs, dim=-1)  # [1]
            action = action.item()  # 转换为标量
            actions.append(action)
        return actions  # 返回长度为 n_agents 的动作列表
