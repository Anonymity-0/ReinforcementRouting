import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from ppo_model import PPOAgent
from mappo_model import MAPPOAgent
from config import *
import time
import random
from collections import defaultdict
import os
import matplotlib.pyplot as plt

def train_and_evaluate(algo_name='dqn', train_episodes=NUM_EPISODES, eval_episodes=100):
    """训练并评估指定的算法
    
    Args:
        algo_name: 算法名称 ('dqn', 'ppo', 或 'mappo')
        train_episodes: 训练回合数
        eval_episodes: 评估回合数
    """
    print(f"\n开始训练和评估 {algo_name.upper()} 模型...")
    
    # 创建保存目录
    os.makedirs(f'models/{algo_name}', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 训练阶段
    train_results = train_agent(algo_name, train_episodes)
    
    # 评估阶段
    eval_results = evaluate_agent(algo_name, eval_episodes)
    
    # 保存结果
    save_results(algo_name, train_results, eval_results)
    
    # 绘制训练过程图
    plot_training_curves(algo_name, train_results)
    
    return train_results, eval_results

def train_agent(algo_name, num_episodes):
    """训练智能体"""
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 初始化智能体
    if algo_name == 'dqn':
        agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    elif algo_name == 'ppo':
        agent = PPOAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    else:  # mappo
        agent = MAPPOAgent(state_size, action_size, env.n_agents, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 训练统计
    stats = {
        'episode_rewards': [],
        'path_lengths': [],
        'delays': [],
        'bandwidths': [],
        'losses': [],
        'success_rate': []
    }
    
    for episode in range(num_episodes):
        episode_stats = run_episode(env, agent, algo_name, training=True)
        
        # 更新统计信息
        for key in episode_stats:
            if key in stats:
                stats[key].append(episode_stats[key])
        
        # 每个episode都输出训练进度
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"奖励: {episode_stats['episode_reward']:.2f}")
        print(f"路径长度: {episode_stats['path_length']}")
        print(f"平均延迟: {episode_stats['delay']:.2f} ms")
        print(f"平均带宽: {episode_stats['bandwidth']:.2f} MHz")
        print(f"平均丢包率: {episode_stats['loss']:.2f}%")
        print(f"是否成功: {'是' if episode_stats['success'] else '否'}")
        print(f"路径: {' -> '.join(episode_stats['path'])}")
        
        # 保存检查点
        if (episode + 1) % 100 == 0:
            save_checkpoint(agent, algo_name, episode + 1, stats)
    
    # 保存最终模型
    save_final_model(agent, algo_name)
    
    return stats

def evaluate_agent(algo_name, num_episodes):
    """评估智能体"""
    env = SatelliteEnv()
    model_path = f'models/{algo_name}/final_model.pth'
    
    # 加载训练好的模型
    state_size, action_size = env.reset()
    if algo_name == 'dqn':
        agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.epsilon = 0  # 评估时不使用探索
    elif algo_name == 'ppo':
        agent = PPOAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
        agent.actor_critic.load_state_dict(torch.load(model_path))
    else:  # mappo
        agent = MAPPOAgent(state_size, action_size, env.n_agents, env.get_leo_names(), env.get_leo_to_meo_mapping())
        agent.network.load_state_dict(torch.load(model_path))
    
    # 评估统计
    stats = defaultdict(list)
    
    for episode in range(num_episodes):
        episode_stats = run_episode(env, agent, algo_name, training=False)
        for key, value in episode_stats.items():
            stats[key].append(value)
    
    return stats

def run_episode(env, agent, algo_name, training=True):
    """运行一个episode"""
    env.reset()
    
    # 随机选择源节点和目标节点
    all_leos = list(env.leo_nodes.keys())
    source = random.choice(all_leos)
    destination = random.choice([leo for leo in all_leos if leo != source])
    
    path = [source]
    total_reward = 0
    metrics_history = defaultdict(list)
    current_leo = source
    
    while len(path) < MAX_PATH_LENGTH:
        # 获取状态和动作
        if algo_name == 'mappo':
            state = agent.get_states(env, [current_leo], [destination])
            available_actions = [env.get_available_actions(current_leo)]
            action = agent.choose_actions(state, available_actions, env, 
                                       [current_leo], [destination], [path])[0]
        else:
            state = agent.get_state(env, current_leo, destination)
            available_actions = env.get_available_actions(current_leo)
            action = agent.choose_action(state, available_actions, env,
                                      current_leo, destination, path)
        
        if action is None:
            break
        
        # 执行动作
        next_state, reward, done, info = env.step(current_leo, action, path)
        
        # 记录性能指标
        metrics = info.get('link_stats', {})
        for key in ['delay', 'bandwidth', 'loss']:
            metrics_history[key].append(metrics.get(key, 0))
        
        # 训练更新
        if training:
            if algo_name == 'dqn':
                agent.memorize(state, action, reward, next_state, done)
                agent.replay(BATCH_SIZE)
            elif algo_name == 'ppo':
                agent.rewards.append(reward)
                agent.masks.append(1 - done)
            else:  # mappo
                for i in range(agent.n_agents):
                    agent.rewards[i].append(reward)
                    agent.masks[i].append(1 - done)
        
        total_reward += reward
        current_leo = list(env.leo_nodes.keys())[action]
        path.append(current_leo)
        
        if done or current_leo == destination:
            break
    
    # 返回episode统计信息
    return {
        'episode_reward': total_reward,
        'path_length': len(path),
        'delay': np.mean(metrics_history['delay']),
        'bandwidth': np.mean(metrics_history['bandwidth']),
        'loss': np.mean(metrics_history['loss']),
        'success': current_leo == destination,
        'path': path
    }

def print_progress(episode, total_episodes, stats):
    """打印训练进度"""
    recent = lambda x: x[-100:] if len(x) >= 100 else x
    print(f"\nEpisode {episode}/{total_episodes}")
    print(f"最近100回合平均奖励: {np.mean(recent(stats['episode_rewards'])):.2f}")
    print(f"最近100回合平均路径长度: {np.mean(recent(stats['path_lengths'])):.2f}")
    print(f"最近100回合平均延迟: {np.mean(recent(stats['delays'])):.2f} ms")
    print(f"最近100回合平均带宽: {np.mean(recent(stats['bandwidths'])):.2f} MHz")
    print(f"最近100回合平均丢包率: {np.mean(recent(stats['losses'])):.2f}%")
    print(f"最近100回合成功率: {np.mean(recent(stats['success_rate']))*100:.2f}%")

def save_checkpoint(agent, algo_name, episode, stats):
    """保存检查点"""
    checkpoint = {
        'episode': episode,
        'stats': stats
    }
    
    if algo_name == 'dqn':
        checkpoint.update({
            'model_state_dict': agent.policy_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        })
    elif algo_name == 'ppo':
        checkpoint.update({
            'model_state_dict': agent.actor_critic.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict()
        })
    else:  # mappo
        checkpoint.update({
            'model_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict()
        })
    
    torch.save(checkpoint, f'models/{algo_name}/checkpoint_episode_{episode}.pth')

def save_final_model(agent, algo_name):
    """保存最终模型"""
    if algo_name == 'dqn':
        torch.save(agent.policy_net.state_dict(), f'models/{algo_name}/final_model.pth')
    elif algo_name == 'ppo':
        torch.save(agent.actor_critic.state_dict(), f'models/{algo_name}/final_model.pth')
    else:  # mappo
        torch.save(agent.network.state_dict(), f'models/{algo_name}/final_model.pth')

def save_results(algo_name, train_results, eval_results):
    """保存训练和评估结果"""
    results = {
        'train': train_results,
        'eval': eval_results
    }
    torch.save(results, f'results/{algo_name}_results.pth')

def plot_training_curves(algo_name, stats):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    metrics = ['episode_rewards', 'path_lengths', 'delays', 
              'bandwidths', 'losses', 'success_rate']
    titles = ['Episode Rewards', 'Path Lengths', 'Average Delay',
              'Average Bandwidth', 'Loss Rate', 'Success Rate']
    
    for ax, metric, title in zip(axes, metrics, titles):
        ax.plot(stats[metric])
        ax.set_title(title)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{algo_name}_training_curves.png')
    plt.close()

if __name__ == "__main__":
    # 训练并评估所有算法
    algorithms = ['dqn', 'ppo', 'mappo']
    results = {}
    
    for algo in algorithms:
        print(f"\n开始训练和评估 {algo.upper()} 算法...")
        train_stats, eval_stats = train_and_evaluate(algo)
        results[algo] = {
            'train': train_stats,
            'eval': eval_stats
        }
    
    # 保存所有结果
    torch.save(results, 'results/all_algorithms_results.pth')
