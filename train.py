# train.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from env import SatelliteEnv
from agents.qmix_agent import QMIXAgent
from agents.mappo_agent import MAPPOAgent
from agents.tarmac_agent import TarMACAgent

def train(agent_name, episodes=1000, steps=50, batch_size=128, gamma=0.99, learning_rate=0.001):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化环境
    env = SatelliteEnv(service_type='delay_sensitive')
    n_agents = env.n_meo
    obs_sizes = [len(obs) for obs in env.get_observation()]

    # 根据 agent_name 初始化智能体
    if agent_name == 'qmix':
        agent = QMIXAgent(n_agents=n_agents, obs_sizes=obs_sizes, action_dim=env.k_paths,
                          hidden_dim=64, learning_rate=learning_rate, device=device)
    elif agent_name == 'mappo':
        agent = MAPPOAgent(n_agents=n_agents, obs_sizes=obs_sizes, action_dim=env.k_paths,
                           hidden_dim=64, learning_rate=learning_rate, device=device)
    elif agent_name == 'tarmac':
        comm_dim = 16
        num_heads = 2
        agent = TarMACAgent(n_agents=n_agents, obs_sizes=obs_sizes, action_dim=env.k_paths,
                            hidden_dim=64, comm_dim=comm_dim, num_heads=num_heads,
                            learning_rate=learning_rate, device=device)
    else:
        raise ValueError(f"未知的智能体名称：{agent_name}")

    epsilon = 1.0
    rewards = []

    for episode in range(episodes):
        observations = env.reset()
        episode_reward = 0
        for step in range(steps):
            # 根据算法选择动作
            if agent_name == 'tarmac':
                # TarMAC 特殊处理 epsilon-greedy
                if np.random.rand() < epsilon:
                    actions = [np.random.choice(env.k_paths) for _ in range(n_agents)]
                else:
                    actions = agent.select_action(observations)
            else:
                actions = agent.select_action(observations, epsilon)

            next_observations, reward, done = env.step(actions)
            agent.store_experience((observations, actions, reward, next_observations, done))

            if agent_name != 'tarmac':
                agent.update(batch_size, gamma)

            observations = next_observations
            episode_reward += reward
            if done:
                break

        if agent_name == 'tarmac':
            agent.update(batch_size, gamma)

        epsilon = max(0.05, epsilon * 0.995)
        rewards.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward:.4f}, Epsilon: {epsilon:.4f}")

        # 每100个回合保存一次模型
        if (episode + 1) % 100 == 0:
            model_path = f"{agent_name}_model_episode_{episode+1}.pth"
            if agent_name == 'qmix':
                agent.save(model_path)
            else:
                torch.save(agent.network.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")

    # 最后保存一次模型
    model_path = f"{agent_name}_model_final.pth"
    if agent_name == 'qmix':
        agent.save(model_path)
    else:
        torch.save(agent.network.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # 绘制奖励曲线
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{agent_name.upper()} Training Reward')
    plt.show()

    return agent  # 返回训练好的智能体

# 如果您希望在 Notebook 中运行，可以直接调用 train 函数，例如：
# agent = train('qmix')
