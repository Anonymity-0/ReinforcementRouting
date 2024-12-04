from env import SatelliteEnv


def inference_traditional(routing_algorithm='dijkstra', num_steps=50):
    env = SatelliteEnv(service_type='delay_sensitive')
    observation = env.reset()
    total_reward = 0
    qos_metrics = []

    for step in range(num_steps):
        if routing_algorithm == 'dijkstra':
            action = env.dijkstra_routing()
        elif routing_algorithm == 'shortest_path':
            action = env.shortest_path_routing()
        else:
            raise ValueError(f"未知的路由算法：{routing_algorithm}")

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

    print(f"Total Reward: {total_reward}")
    avg_delay = sum(m['delay'] for m in qos_metrics) / len(qos_metrics)
    avg_loss = sum(m['packet_loss'] for m in qos_metrics) / len(qos_metrics)
    avg_delivery = sum(m['delivery_rate'] for m in qos_metrics) / len(qos_metrics)
    print(f"Average Delay: {avg_delay}")
    print(f"Average Packet Loss: {avg_loss}")
    print(f"Average Delivery Rate: {avg_delivery}")

    return qos_metrics