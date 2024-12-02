# inference.py

import torch
from env import SatelliteEnv
from agents.qmix_agent import QMIXAgent
from agents.mappo_agent import MAPPOAgent
from agents.tarmac_agent import TarMACAgent

def inference(agent_name, model_path, num_steps=50):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化环境
    env = SatelliteEnv(service_type='delay_sensitive')
    obs_sizes = [len(obs) for obs in env.get_observation()]
    n_agents = env.n_meo

    # 根据 agent_name 初始化智能体
    if agent_name == 'qmix':
        agent = QMIXAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=0.001,
            device=device
        )
        # 加载模型
        agent.load(model_path)
    elif agent_name == 'mappo':
        agent = MAPPOAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=0.001,
            device=device
        )
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        agent.critic.load_state_dict(checkpoint['critic'])
        for i, actor in enumerate(agent.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        print(f"模型已从 {model_path} 加载")
    elif agent_name == 'tarmac':
        comm_dim = 16  # 通信向量的维度，与训练时一致
        num_heads = 2
        agent = TarMACAgent(
            n_agents=n_agents,
            obs_sizes=obs_sizes,
            action_dim=env.k_paths,
            hidden_dim=64,
            comm_dim=comm_dim,
            num_heads=num_heads,
            learning_rate=0.001,
            device=device
        )
        # 加载模型
        agent.network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        raise ValueError(f"未知的智能体名称：{agent_name}")

    epsilon = 0.0  # 设为0，始终选择最佳动作

    observations = env.reset()
    total_reward = 0
    qos_metrics = []

    for step in range(num_steps):
        if agent_name == 'tarmac':
            actions = agent.select_action(observations)
        else:
            actions = agent.select_action(observations, epsilon)

        next_observations, reward, done = env.step(actions)
        observations = next_observations
        total_reward += reward

        # 计算QoS指标
        candidate_paths = env.get_candidate_paths(env.src, env.dst)
        if not candidate_paths:
            break
        path = candidate_paths[actions[0]]  # 假设所有智能体选择相同的路径
        delay, loss, delivery = env.calculate_qos_metrics(path)
        qos_metrics.append({
            'delay': delay,
            'packet_loss': loss,
            'delivery_rate': delivery
        })

        print(f"Step {step+1}:")
        print(f"  Actions: {actions}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Delay: {delay:.4f}s")
        print(f"  Packet Loss: {loss:.4f}")
        print(f"  Delivery Rate: {delivery:.4f}")

        if done:
            break

    print(f"\nInference Results:")
    print(f"Total Reward: {total_reward:.4f}")

    # 计算平均QoS指标
    avg_delay = sum(m['delay'] for m in qos_metrics) / len(qos_metrics)
    avg_loss = sum(m['packet_loss'] for m in qos_metrics) / len(qos_metrics)
    avg_delivery = sum(m['delivery_rate'] for m in qos_metrics) / len(qos_metrics)

    print(f"Average Delay: {avg_delay:.4f}s")
    print(f"Average Packet Loss: {avg_loss:.4f}")
    print(f"Average Delivery Rate: {avg_delivery:.4f}")

    return agent, qos_metrics  # 返回智能体和 QoS 指标，方便在 Notebook 中进一步分析

# 在 Notebook 中可以直接调用 inference 函数，例如：
# agent, qos_metrics = inference('qmix', 'qmix_model_final.pth')
