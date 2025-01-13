import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from config import *
import time
import random
from collections import defaultdict

def train_dqn():
    """训练DQN模型"""
    print("\n开始训练DQN模型...")
    
    # 初始化环境
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 初始化智能体
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 训练统计
    episode_rewards = []
    path_lengths = []
    performance_stats = defaultdict(list)
    
    # 开始训练
    for episode in range(NUM_EPISODES):
        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
        episode_start = time.time()
        
        # 重置环境
        env.reset()
        
        # 随机选择源节点和目标节点
        all_leos = list(env.leo_nodes.keys())
        source = random.choice(all_leos)
        destination = random.choice([leo for leo in all_leos if leo != source])
        
        print(f"路径: {source} -> {destination}")
        print(f"源MEO区域: {env.leo_to_meo[source]}, 目标MEO区域: {env.leo_to_meo[destination]}")
        
        # 初始化路径和奖励
        path = [source]
        total_reward = 0
        current_leo = source
        
        # 记录性能指标
        episode_metrics = {
            'delays': [],
            'bandwidths': [],
            'loss_rates': [],
            'queue_utilizations': []
        }
        
        # 执行一个episode
        while len(path) < MAX_PATH_LENGTH:
            # 获取当前状态
            state = agent.get_state(env, current_leo, destination)
            
            # 获取可用动作
            available_actions = env.get_available_actions(current_leo)
            if not available_actions:
                break
                
            # 选择动作
            action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
            if action is None:
                break
                
            # 执行动作
            next_state, reward, done, info = env.step(current_leo, action, path)
            
            # 记录性能指标
            metrics = info.get('link_stats', {})
            episode_metrics['delays'].append(metrics.get('delay', 0))
            episode_metrics['bandwidths'].append(metrics.get('bandwidth', 0))
            episode_metrics['loss_rates'].append(metrics.get('loss', 0))
            episode_metrics['queue_utilizations'].append(metrics.get('queue_utilization', 0))
            
            # 存储经验
            agent.memorize(state, action, reward, next_state, done)
            
            # 更新状态和奖励
            total_reward += reward
            current_leo = list(env.leo_nodes.keys())[action]
            path.append(current_leo)
            
            # 经验回放
            agent.replay(BATCH_SIZE)
            
            if done or current_leo == destination:
                break
        
        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # 记录统计信息
        episode_rewards.append(total_reward)
        path_lengths.append(len(path))
        
        # 计算平均性能指标
        avg_delay = np.mean(episode_metrics['delays']) if episode_metrics['delays'] else 0
        avg_bandwidth = np.mean(episode_metrics['bandwidths']) if episode_metrics['bandwidths'] else 0
        avg_loss = np.mean(episode_metrics['loss_rates']) if episode_metrics['loss_rates'] else 0
        avg_queue = np.mean(episode_metrics['queue_utilizations']) if episode_metrics['queue_utilizations'] else 0
        
        performance_stats['delay'].append(avg_delay)
        performance_stats['bandwidth'].append(avg_bandwidth)
        performance_stats['loss'].append(avg_loss)
        performance_stats['queue'].append(avg_queue)
        
        # 输出episode信息
        episode_duration = time.time() - episode_start
        print(f"\n路径详情:")
        print(f"完整路径: {' -> '.join(path)}")
        print(f"路径长度: {len(path)}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"探索率: {agent.epsilon:.4f}")
        
        print("\n性能指标:")
        print(f"平均延迟: {avg_delay:.2f} ms")
        print(f"平均带宽: {avg_bandwidth:.2f} MHz")
        print(f"平均丢包率: {avg_loss:.2f}%")
        print(f"平均队列利用率: {avg_queue:.2f}%")
        
        # 每100个episode输出统计信息
        if (episode + 1) % 100 == 0:
            print("\n阶段性统计:")
            print(f"最近100个episode平均奖励: {np.mean(episode_rewards[-100:]):.2f}")
            print(f"最近100个episode平均路径长度: {np.mean(path_lengths[-100:]):.2f}")
            print(f"最近100个episode平均延迟: {np.mean(performance_stats['delay'][-100:]):.2f} ms")
            print(f"最近100个episode平均带宽: {np.mean(performance_stats['bandwidth'][-100:]):.2f} MHz")
            print(f"最近100个episode平均丢包率: {np.mean(performance_stats['loss'][-100:]):.2f}%")
            
            # 保存检查点
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'rewards': episode_rewards,
                'performance_stats': dict(performance_stats)
            }, f'models/checkpoint_episode_{episode+1}.pth')
    
    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), 'models/final_model.pth')
    
    return episode_rewards, performance_stats

def evaluate_model(model_path):
    """评估训练好的模型"""
    print("\n开始评估模型...")
    
    # 初始化环境和智能体
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 加载模型
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # 评估时不使用探索
    
    # 评估统计
    test_episodes = 100
    success_count = 0
    total_rewards = []
    path_lengths = []
    performance_metrics = defaultdict(list)
    
    for episode in range(test_episodes):
        env.reset()
        
        # 随机选择源和目标
        all_leos = list(env.leo_nodes.keys())
        source = random.choice(all_leos)
        destination = random.choice([leo for leo in all_leos if leo != source])
        
        path = [source]
        total_reward = 0
        current_leo = source
        
        # 记录性能指标
        episode_metrics = {
            'delays': [],
            'bandwidths': [],
            'loss_rates': [],
            'queue_utilizations': []
        }
        
        while len(path) < MAX_PATH_LENGTH:
            state = agent.get_state(env, current_leo, destination)
            available_actions = env.get_available_actions(current_leo)
            
            if not available_actions:
                break
                
            action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
            if action is None:
                break
                
            next_state, reward, done, info = env.step(current_leo, action, path)
            
            # 记录性能指标
            metrics = info.get('link_stats', {})
            episode_metrics['delays'].append(metrics.get('delay', 0))
            episode_metrics['bandwidths'].append(metrics.get('bandwidth', 0))
            episode_metrics['loss_rates'].append(metrics.get('loss', 0))
            episode_metrics['queue_utilizations'].append(metrics.get('queue_utilization', 0))
            
            total_reward += reward
            current_leo = list(env.leo_nodes.keys())[action]
            path.append(current_leo)
            
            if done or current_leo == destination:
                if current_leo == destination:
                    success_count += 1
                break
        
        # 记录统计信息
        total_rewards.append(total_reward)
        path_lengths.append(len(path))
        
        # 计算平均性能指标
        for metric_name, values in episode_metrics.items():
            if values:
                performance_metrics[metric_name].append(np.mean(values))
        
        # 输出每个episode的结果
        print(f"\nTest Episode {episode + 1}:")
        print(f"路径: {' -> '.join(path)}")
        print(f"长度: {len(path)}")
        print(f"奖励: {total_reward:.2f}")
        print(f"平均延迟: {np.mean(episode_metrics['delays']):.2f} ms")
        print(f"平均带宽: {np.mean(episode_metrics['bandwidths']):.2f} MHz")
        print(f"平均丢包率: {np.mean(episode_metrics['loss_rates']):.2f}%")
    
    # 输出总体评估结果
    print("\n评估结果总结:")
    print(f"成功率: {success_count/test_episodes*100:.2f}%")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"平均路径长度: {np.mean(path_lengths):.2f}")
    print(f"平均延迟: {np.mean(performance_metrics['delays']):.2f} ms")
    print(f"平均带宽: {np.mean(performance_metrics['bandwidths']):.2f} MHz")
    print(f"平均丢包率: {np.mean(performance_metrics['loss_rates']):.2f}%")
