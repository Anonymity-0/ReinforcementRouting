# train_dqn.py

from agents.dqn_agent import DQNAgent
from env import SatelliteEnv


def train_dqn(episodes=1000, steps=50, batch_size=128, gamma=0.99, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SatelliteEnv(service_type='delay_sensitive')
    obs_size = len(env.get_observation())
    action_dim = env.k_paths
    agent = DQNAgent(obs_size, action_dim, hidden_dim=64, learning_rate=learning_rate, device=device)
    epsilon = 1.0
    rewards = []
    for episode in range(episodes):
        observation = env.reset()
        episode_reward = 0
        for step in range(steps):
            action = agent.select_action(observation, epsilon)
            next_observation, reward, done, _ = env.step(action)
            agent.store_experience((observation, action, reward, next_observation, done))
            agent.update(batch_size, gamma)
            observation = next_observation
            episode_reward += reward
            if done:
                break
        agent.update_target_network()
        epsilon = max(epsilon * 0.995, 0.01)
        rewards.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}")
    # 保存模型
    torch.save(agent.policy_net.state_dict(), 'dqn_model.pth')