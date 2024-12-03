# inference.py

import torch
from env import SatelliteEnv
from agents.qmix_agent import QMIXAgent
from agents.mappo_agent import MAPPOAgent
from agents.tarmac_agent import TarMACAgent
from agents.dqn_agent import DQNAgent
import pandas as pd
import matplotlib.pyplot as plt

def inference(agent_name, model_path, num_steps=50):
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
    elif agent_name == 'dqn':
        agent = DQNAgent(
            obs_size=obs_sizes[0],
            action_dim=env.k_paths,
            hidden_dim=64,
            learning_rate=0.001,
            device=device
        )
        # 加载模型
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"未知的智能体名称：{agent_name}")

    epsilon = 0.0  # 设为0，始终选择最佳动作

    observations = env.reset()
    total_reward = 0
    qos_metrics = []

    for step in range(num_steps):
        if agent_name == 'tarmac':
            actions = agent.select_action(observations)
        elif agent_name == 'dqn':
            actions = agent.select_action(observations, epsilon)
        else:
            actions = agent.select_action(observations, epsilon)

        next_observations, reward, done, _ = env.step(actions)
        observations = next_observations
        total_reward += reward

        # 计算QoS指标
        candidate_paths = env.get_candidate_paths(env.src, env.dst)
        if not candidate_paths:
            break
        if agent_name == 'dqn':
            path = candidate_paths[actions]
        else:
            path = candidate_paths[actions[0]]  # 假设所有智能体选择相同的路径
        delay, loss, delivery = env.calculate_qos_metrics(path)
        qos_metrics.append({
            'delay': delay,
            'packet_loss': loss,
            'delivery_rate': delivery,
        })

        if done:
            break

    # 输出结果
    print(f"Total Reward: {total_reward}")
    avg_delay = sum(m['delay'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    avg_loss = sum(m['packet_loss'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    avg_delivery = sum(m['delivery_rate'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    print(f"Average Delay: {avg_delay}s")
    print(f"Average Packet Loss: {avg_loss}")
    print(f"Average Delivery Rate: {avg_delivery}")

    return agent, qos_metrics

def inference_traditional(routing_algorithm='dijkstra', num_steps=50):
    env = SatelliteEnv(service_type='delay_sensitive', multi_agent=False)
    observations = env.reset()
    total_reward = 0
    qos_metrics = []

    for step in range(num_steps):
        if routing_algorithm == 'dijkstra':
            path = env.get_shortest_path(env.src, env.dst)
        else:
            raise ValueError(f"未知的路由算法：{routing_algorithm}")

        next_observations, reward, done, _ = env.step(path)
        observations = next_observations
        total_reward += reward

        # 计算QoS指标
        delay, loss, delivery = env.calculate_qos_metrics(path)
        qos_metrics.append({
            'delay': delay,
            'packet_loss': loss,
            'delivery_rate': delivery,
        })

        if done:
            break

    # 输出结果
    print(f"Total Reward (Traditional): {total_reward}")
    avg_delay = sum(m['delay'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    avg_loss = sum(m['packet_loss'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    avg_delivery = sum(m['delivery_rate'] for m in qos_metrics) / len(qos_metrics) if qos_metrics else 0
    print(f"Average Delay (Traditional): {avg_delay}s")
    print(f"Average Packet Loss (Traditional): {avg_loss}")
    print(f"Average Delivery Rate (Traditional): {avg_delivery}")

    return qos_metrics

# 如果您希望在命令行中运行，可以使用以下代码
# if __name__ == "__main__":
#     # 示例：推理 QMIX 算法
#     agent_name = 'qmix'
#     model_path = 'qmix_model_final.pth'
#     agent, qos_metrics = inference(agent_name, model_path)

#     # 您也可以推理其他算法，例如 DQN
#     # agent_name = 'dqn'
#     # model_path = 'dqn_model_final.pth'
#     # agent, qos_metrics = inference(agent_name, model_path)

#     # 传统算法推理
#     dijkstra_qos_metrics = inference_traditional()

#     # 数据分析
#     qmix_df = pd.DataFrame(qos_metrics)
#     dijkstra_df = pd.DataFrame(dijkstra_qos_metrics)

#     # 绘制对比图
#     plt.figure(figsize=(12, 6))
#     if not qmix_df.empty:
#         plt.plot(qmix_df['delay'], label=f'{agent_name.upper()} 算法')
#     if not dijkstra_df.empty:
#         plt.plot(dijkstra_df['delay'], label='Dijkstra 算法')

#     plt.xlabel('步骤')
#     plt.ylabel('时延 (秒)')
#     plt.title('时延对比')
#     plt.legend()
#     plt.show()

#     # 输出平均 QoS 指标
#     if not qmix_df.empty:
#         print(f"{agent_name.upper()} 算法平均时延:", qmix_df['delay'].mean())
#     if not dijkstra_df.empty:
#         print("Dijkstra 算法平均时延:", dijkstra_df['delay'].mean())