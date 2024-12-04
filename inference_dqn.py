# inference_dqn.py

def inference_dqn(model_path, num_steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SatelliteEnv(service_type='delay_sensitive')
    obs_size = len(env.get_observation())
    action_dim = env.k_paths
    agent = DQNAgent(obs_size, action_dim, hidden_dim=64, learning_rate=0.001, device=device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    observation = env.reset()
    total_reward = 0
    qos_metrics = []
    for step in range(num_steps):
        action = agent.select_action(observation, epsilon=0.0)
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation
        total_reward += reward
        # 计算 QoS 指标
        candidate_paths = env.get_candidate_paths(env.src, env.dst)
        path = candidate_paths[action]
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
    avg_delay = sum(m['delay'] for m in qos_metrics) / len(qos_metrics)
    avg_loss = sum(m['packet_loss'] for m in qos_metrics) / len(qos_metrics)
    avg_delivery = sum(m['delivery_rate'] for m in qos_metrics) / len(qos_metrics)
    print(f"Average Delay: {avg_delay}")
    print(f"Average Packet Loss: {avg_loss}")
    print(f"Average Delivery Rate: {avg_delivery}")