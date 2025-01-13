import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from config import *
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import os

def train_dqn():
    """训练DQN代理"""
    # 确保模型保存目录存在
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print("初始化环境...")
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    print(f"状态空间大小: {state_size}")
    print(f"动作空间大小: {action_size}")
    
    # 初始化DQN代理
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 训练统计
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    # 添加网络指标统计
    network_metrics = {
        'avg_delay': [],
        'avg_bandwidth_util': [],
        'avg_loss_rate': [],
        'avg_queue_util': []
    }
    
    print("\n开始训练...")
    try:
        for episode in range(NUM_EPISODES):
            total_reward = 0
            episode_metrics = {
                'delays': [],
                'bandwidth_utils': [],
                'loss_rates': [],
                'queue_utils': []
            }
            
            state_size, action_size = env.reset()
            
            # 随机选择源节点和目标节点
            all_leos = env.get_leo_names()
            source = random.choice(all_leos)
            destination = random.choice([leo for leo in all_leos if leo != source])
            
            path = [source]
            current_leo = source
            step = 0
            
            while step < MAX_PATH_LENGTH:
                # 获取当前状态
                state = agent.get_state(env, current_leo, destination)
                
                # 获取可用动作
                available_actions = env.get_available_actions(current_leo)
                
                # 选择动作
                action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
                
                if action is None:
                    break
                    
                # 执行动作
                next_leo = env.get_leo_names()[action]
                next_state, reward, done, info = env.step(current_leo, action, path)
                
                # 存储经验
                agent.memorize(state, action, reward, next_state, done)
                
                # 经验回放
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
                
                total_reward += reward
                current_leo = next_leo
                path.append(current_leo)
                
                if done or current_leo == destination:
                    break
                    
                step += 1
                
                # 收集当前链路的性能指标
                if info and 'link_stats' in info:
                    stats = info['link_stats']
                    episode_metrics['delays'].append(stats['delay'])
                    episode_metrics['bandwidth_utils'].append(stats['bandwidth_utilization'])
                    episode_metrics['loss_rates'].append(stats['loss'])
                    episode_metrics['queue_utils'].append(stats['queue_utilization'])
                
            # 计算并记录本回合的平均指标
            if episode_metrics['delays']:
                network_metrics['avg_delay'].append(np.mean(episode_metrics['delays']))
                network_metrics['avg_bandwidth_util'].append(np.mean(episode_metrics['bandwidth_utils']))
                network_metrics['avg_loss_rate'].append(np.mean(episode_metrics['loss_rates']))
                network_metrics['avg_queue_util'].append(np.mean(episode_metrics['queue_utils']))
            
            # 更新目标网络
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            # 记录统计信息
            episode_rewards.append(total_reward)
            avg_rewards.append(total_reward)
            avg_reward = np.mean(avg_rewards)
            
            # 每个episode结束时打印汇总信息
            print(f"\nEpisode {episode}/{NUM_EPISODES}")
            source_meo = env.leo_to_meo[source]
            dest_meo = env.leo_to_meo[destination]
            print(f"源节点: {source}(MEO区域: {source_meo}) -> 目标节点: {destination}(MEO区域: {dest_meo})")
            
            # 打印完整路径及其MEO区域信息
            path_info = []
            for leo in path:
                meo = env.leo_to_meo[leo]
                path_info.append(f"{leo}({meo})")
            print(f"路径: {' -> '.join(path_info)}")
            
            print(f"总奖励: {total_reward:.2f}")
            print(f"平均奖励: {np.mean(avg_rewards):.2f}")
            print(f"探索率: {agent.epsilon:.4f}")
            
            # 打印网络性能指标
            if episode_metrics['delays']:
                print("\n网络性能指标:")
                print(f"平均延迟: {np.mean(episode_metrics['delays']):.2f} ms")
                print(f"平均带宽利用率: {np.mean(episode_metrics['bandwidth_utils']):.2f}%")
                print(f"平均丢包率: {np.mean(episode_metrics['loss_rates']):.2f}%")
                print(f"平均队列利用率: {np.mean(episode_metrics['queue_utils']):.2f}%")
                
                # 添加MEO区域切换统计
                meo_switches = sum(1 for i in range(len(path)-1) 
                                 if env.leo_to_meo[path[i]] != env.leo_to_meo[path[i+1]])
                print(f"MEO区域切换次数: {meo_switches}")
            
            # 构建path_stats字典
            path_stats = {
                'sent': info.get('packets_sent', []),
                'received': info.get('packets_received', []),
                'dropped': info.get('packets_dropped', []),
                'lost': info.get('packets_lost', [])
            }
            
            # 构建metrics字典
            metrics = {
                'delay': np.mean(episode_metrics['delays']) if episode_metrics['delays'] else 0,
                'bandwidth': np.mean(episode_metrics['bandwidth_utils']) if episode_metrics['bandwidth_utils'] else 0,
                'rewards': episode_rewards
            }
            
            # 使用print_episode_stats打印详细统计信息
            print_episode_stats(episode, NUM_EPISODES, path, path_stats, metrics, agent, env)
        
        # 训练结束后绘制性能指标曲线
        plt.figure(figsize=(15, 10))
        
        # 绘制奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.title('训练奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # 绘制延迟曲线
        plt.subplot(2, 2, 2)
        plt.plot(network_metrics['avg_delay'])
        plt.title('平均延迟变化')
        plt.xlabel('Episode')
        plt.ylabel('Delay (ms)')
        
        # 绘制带宽利用率曲线
        plt.subplot(2, 2, 3)
        plt.plot(network_metrics['avg_bandwidth_util'])
        plt.title('平均带宽利用率变化')
        plt.xlabel('Episode')
        plt.ylabel('Bandwidth Utilization (%)')
        
        # 绘制丢包率曲线
        plt.subplot(2, 2, 4)
        plt.plot(network_metrics['avg_loss_rate'])
        plt.title('平均丢包率变化')
        plt.xlabel('Episode')
        plt.ylabel('Loss Rate (%)')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        # 保存训练指标数据
        np.save('network_metrics.npy', network_metrics)
        
        print("\n训练完成!")
        print(f"最佳平均奖励: {best_avg_reward:.2f}")
        print("\n最终网络性能:")
        print(f"平均延迟: {np.mean(network_metrics['avg_delay'][-100:]):.2f} ms")
        print(f"平均带宽利用率: {np.mean(network_metrics['avg_bandwidth_util'][-100:])*100:.2f}%")
        print(f"平均丢包率: {np.mean(network_metrics['avg_loss_rate'][-100:]):.2f}%")
        print(f"平均队列利用率: {np.mean(network_metrics['avg_queue_util'][-100:])*100:.2f}%")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
    finally:
        # 保存最终模型
        torch.save(agent.policy_net.state_dict(), os.path.join(model_dir, 'final_model.pth'))
        print("已保存最终模型")

def evaluate_model(model_path, num_episodes=100):
    """评估训练好的模型"""
    print("\n开始评估模型...")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在")
        return
        
    try:
        env = SatelliteEnv()
        state_size, action_size = env.reset()
        
        # 初始化代理并加载模型
        agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.epsilon = 0.0  # 关闭探索
        
        success_count = 0
        total_rewards = []
        path_lengths = []
        
        # 添加性能指标统计
        evaluation_metrics = {
            'delays': [],
            'bandwidth_utils': [],
            'loss_rates': [],
            'queue_utils': []
        }
        
        for episode in range(num_episodes):
            state_size, action_size = env.reset()
            
            # 随机选择源节点和目标节点
            all_leos = env.get_leo_names()
            source = random.choice(all_leos)
            destination = random.choice([leo for leo in all_leos if leo != source])
            
            path = [source]
            current_leo = source
            total_reward = 0
            step = 0
            
            episode_metrics = {
                'delays': [],
                'bandwidth_utils': [],
                'loss_rates': [],
                'queue_utils': []
            }
            
            while step < MAX_PATH_LENGTH:
                state = agent.get_state(env, current_leo, destination)
                available_actions = env.get_available_actions(current_leo)
                action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
                
                if action is None:
                    break
                    
                next_leo = env.get_leo_names()[action]
                next_state, reward, done, info = env.step(current_leo, action, path)
                
                total_reward += reward
                current_leo = next_leo
                path.append(current_leo)
                
                if done or current_leo == destination:
                    if current_leo == destination:
                        success_count += 1
                    break
                    
                step += 1
                
                
    except Exception as e:
        print(f"评估过程出错: {str(e)}")

def print_episode_stats(episode, episodes, path, path_stats, metrics, agent, env):
    """打印每个episode的统计信息"""
    print(f"\n训练轮次 {episode + 1}/{episodes}")
    print(f"路径: {' -> '.join(path)}")
    
    total_sent = len(path_stats['sent'])
    if total_sent > 0:
        # 计算各种包的数量
        dropped_packets = len(path_stats['dropped'])  # 队列丢弃
        lost_packets = len(path_stats['lost'])       # 传输丢失
        received_packets = len(path_stats['received'])
        in_transit_packets = total_sent - (dropped_packets + lost_packets + received_packets)
        
        # 计算总丢包率（包括队列丢弃和传输丢失）
        total_lost = dropped_packets + lost_packets
        total_loss_rate = (total_lost / total_sent) * 100
        
        print(f"\n数据包统计:")
        print(f"  - 总发送包数: {total_sent}")
        print(f"  - 成功接收包数: {received_packets}")
        print(f"  - 总丢失包数: {total_lost}")
        print(f"  - 在途包数: {in_transit_packets}")
        
        print(f"\n性能指标:")
        print(f"  - 总丢包率: {total_loss_rate:.2f}%")
        print(f"  - 传输成功率: {(received_packets/total_sent*100):.2f}%")
        print(f"  - 在途率: {(in_transit_packets/total_sent*100):.2f}%")
        
        # 打印链路性能指标
        if metrics:
            print(f"\n链路指标:")
            print(f"  - 延迟: {metrics['delay']:.2f} ms")
            print(f"  - 带宽: {metrics['bandwidth']:.2f} MHz")
            print(f"  - 丢包率: {total_loss_rate:.2f}%")  # 使用相同的总丢包率
    
    print(f"\n其他信息:")
    print(f"路径长度: {len(path)}")
    print(f"奖励值: {metrics['rewards'][-1]:.2f}")
    print(f"探索率: {agent.epsilon:.3f}")
    print("-" * 50)

if __name__ == "__main__":
    start_time = time.time()
    try:
        # 训练模型
        train_dqn()
        
        # 评估最终模型
        evaluate_model('models/final_model.pth')
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
    finally:
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")