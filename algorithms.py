import numpy as np
from collections import defaultdict
import heapq
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from ppo_model import PPOAgent
from mappo_model import MAPPOAgent

def dijkstra_shortest_path(env, source, destination):
    """Dijkstra最短路径算法"""
    distances = {node: float('infinity') for node in env.leo_nodes}
    distances[source] = 0
    pq = [(0, source)]
    previous = {node: None for node in env.leo_nodes}
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node == destination:
            break
            
        if current_distance > distances[current_node]:
            continue
            
        for neighbor in env.leo_neighbors[current_node]:
            metrics = env._calculate_link_metrics(current_node, neighbor)
            if not metrics:
                continue
                
            distance = current_distance + metrics['delay']
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # 构建路径
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))

def evaluate_algorithm(algo_name, env, num_episodes=100):
    """评估算法性能"""
    results = defaultdict(list)
    
    if algo_name == 'dijkstra':
        for _ in range(num_episodes):
            env.reset()
            # 随机选择源和目标
            all_leos = list(env.leo_nodes.keys())
            source = np.random.choice(all_leos)
            destination = np.random.choice([leo for leo in all_leos if leo != source])
            
            # 使用Dijkstra算法找路径
            path = dijkstra_shortest_path(env, source, destination)
            
            # 计算路径性能指标
            total_delay = 0
            total_bandwidth = float('inf')
            total_loss = 0
            
            for i in range(len(path)-1):
                metrics = env._calculate_link_metrics(path[i], path[i+1])
                if metrics:
                    total_delay += metrics['delay']
                    total_bandwidth = min(total_bandwidth, metrics['bandwidth'])
                    total_loss = max(total_loss, metrics['loss'])
            
            results['path_length'].append(len(path))
            results['delay'].append(total_delay)
            results['bandwidth'].append(total_bandwidth)
            results['loss'].append(total_loss)
            
    else:  # DQN, PPO, MAPPO
        if algo_name == 'dqn':
            agent = DQNAgent(env.state_size, env.action_size, 
                           env.get_leo_names(), env.get_leo_to_meo_mapping())
        elif algo_name == 'ppo':
            agent = PPOAgent(env.state_size, env.action_size,
                           env.get_leo_names(), env.get_leo_to_meo_mapping())
        else:  # MAPPO
            agent = MAPPOAgent(env.state_size, env.action_size, env.n_agents,
                             env.get_leo_names(), env.get_leo_to_meo_mapping())
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            path = []
            
            while not done:
                action = agent.choose_action(state, env.get_available_actions(),
                                          env, path[-1] if path else None)
                if action is None:
                    break
                    
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if 'metrics' in info:
                    results['delay'].append(info['metrics']['delay'])
                    results['bandwidth'].append(info['metrics']['bandwidth'])
                    results['loss'].append(info['metrics']['loss'])
                
                state = next_state
                
            results['path_length'].append(len(path))
            results['reward'].append(episode_reward)
    
    return results

def compare_algorithms():
    """比较不同算法的性能"""
    env = SatelliteEnv()
    algorithms = ['dijkstra', 'dqn', 'ppo', 'mappo']
    all_results = {}
    
    for algo in algorithms:
        print(f"\n评估 {algo.upper()} 算法...")
        results = evaluate_algorithm(algo, env)
        all_results[algo] = results
        
        print(f"\n{algo.upper()} 性能指标:")
        print(f"平均路径长度: {np.mean(results['path_length']):.2f}")
        print(f"平均延迟: {np.mean(results['delay']):.2f} ms")
        print(f"平均带宽: {np.mean(results['bandwidth']):.2f} MHz")
        print(f"平均丢包率: {np.mean(results['loss']):.2f}%")
        if 'reward' in results:
            print(f"平均奖励: {np.mean(results['reward']):.2f}")
    
    return all_results 