# train.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from env import SatelliteEnv
from agents.qmix_agent import QMIXAgent
from agents.mappo_agent import MAPPOAgent
from agents.tarmac_agent import TarMACAgent
from agents.dqn_agent import DQNAgent

def train(agent_name, episodes=1000, steps=50, batch_size=128, gamma=0.99, learning_rate=0.001, save_interval=100):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化环境
    if agent_name == 'dqn':
        env = SatelliteEnv(service_type='delay_sensitive', multi_agent=False)
    else:
        env = SatelliteEnv(service_type='delay_sensitive', multi_agent=True)

    n_agents = env.n_meo
    obs = env.get_observation()
    if isinstance(obs, (list, tuple)):
        obs_sizes = [len(o) for o in obs]
    else:
        obs_sizes = [len(obs)]

    # 根据 agent_name 初始化智能体
    if agent_name == 'qmix':
        agent = QMIXAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=learning_rate,
            device=device
        )
    elif agent_name == 'mappo':
        agent = MAPPOAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=learning_rate,
            device=device
        )
    elif agent_name == 'tarmac':
        comm_dim = 16
        num_heads = 2
        agent = TarMACAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            comm_dim=comm_dim,
            num_heads=num_heads,
            learning_rate=learning_rate,
            device=device
        )
    elif agent_name == 'dqn':
        agent = DQNAgent(
            obs_size=obs_sizes[0],
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=learning_rate,
            device=device
        )
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
            elif agent_name == 'dqn':
                actions = agent.select_action(observations, epsilon)
            else:
                actions = agent.select_action(observations, epsilon)

            next_observations, reward, done, _ = env.step(actions)
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

        # 每隔 save_interval 轮保存模型
        if (episode + 1) % save_interval == 0:
            model_path = f"{agent_name}_model_episode_{episode+1}.pth"
            save_model(agent, agent_name, model_path)
            print(f"模型已保存到 {model_path}")

    # 最后保存一次模型
    model_path = f"{agent_name}_model_final.pth"
    save_model(agent, agent_name, model_path)
    print(f"模型已保存到 {model_path}")

    # 绘制奖励曲线
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{agent_name.upper()} Training Reward')
    plt.show()

    return agent  # 返回训练好的智能体

def save_model(agent, agent_name, model_path):
    if agent_name == 'qmix':
        agent.save(model_path)
    elif agent_name == 'mappo':
        checkpoint = {
            'critic': agent.critic.state_dict(),
            'actors': [actor.state_dict() for actor in agent.actors]
        }
        torch.save(checkpoint, model_path)
    elif agent_name == 'tarmac':
        torch.save(agent.network.state_dict(), model_path)
    elif agent_name == 'dqn':
        torch.save(agent.policy_net.state_dict(), model_path)

# # 如果您希望在命令行中运行，可以使用以下代码
# if __name__ == "__main__":
#     # 示例：训练 QMIX 算法
#     train('qmix', episodes=1000)
#     # 其他算法可以类似地调用
#     # train('mappo', episodes=1000)
#     # train('tarmac', episodes=1000)
#     # train('dqn', episodes=1000)