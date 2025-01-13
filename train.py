import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import json
from satellite_env import SatelliteEnv
from config import MAX_PATH_LENGTH
from dqn_model import DQNAgent

def plot_metrics(metrics, save_dir='training_plots'):
    """绘制训练指标图表"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
    
    episodes = range(1, len(metrics['delays']) + 1)
    
    # 延迟
    ax1.plot(episodes, metrics['delays'])
    ax1.set_title('Average Delay (ms)')
    ax1.set_xlabel('Episode')
    ax1.grid(True)
    
    # 带宽
    ax2.plot(episodes, metrics['bandwidths'])
    ax2.set_title('Average Bandwidth (Mbps)')
    ax2.set_xlabel('Episode')
    ax2.grid(True)
    
    # 丢包率
    ax3.plot(episodes, metrics['losses'])
    ax3.set_title('Loss Rate (%)')
    ax3.set_xlabel('Episode')
    ax3.grid(True)
    
    # 路径长度
    ax4.plot(episodes, metrics['path_lengths'])
    ax4.set_title('Path Length')
    ax4.set_xlabel('Episode')
    ax4.grid(True)
    
    # 传输成功率
    ax5.plot(episodes, metrics['transmission_success_rates'])
    ax5.set_title('Transmission Success Rate (%)')
    ax5.set_xlabel('Episode')
    ax5.grid(True)
    
    # 奖励
    ax6.plot(episodes, metrics['rewards'])
    ax6.set_title('Rewards')
    ax6.set_xlabel('Episode')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics_{timestamp}.png')
    plt.close()

def save_training_history(metrics, save_dir='training_history'):
    """保存训练历史数据"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{save_dir}/training_history_{timestamp}.json'
    
    # 将numpy数组转换为列表
    history = {
        'delays': [float(x) for x in metrics['delays']],
        'bandwidths': [float(x) for x in metrics['bandwidths']],
        'losses': [float(x) for x in metrics['losses']],
        'path_lengths': [int(x) for x in metrics['path_lengths']],
        'transmission_success_rates': [float(x) for x in metrics['transmission_success_rates']],
        'rewards': [float(x) for x in metrics['rewards']]
    }
    
    with open(filename, 'w') as f:
        json.dump(history, f, indent=4)

def print_episode_stats(episode, episodes, path, path_stats, metrics, agent, env):
    """打印每个episode的统计信息"""
    print(f"\n训练轮次 {episode + 1}/{episodes}")
    print(f"路径: {' -> '.join(path)}")
    
    total_sent = len(path_stats['sent'])
    if total_sent > 0:
        # 计算各种包的数量
        dropped_packets = len(path_stats['dropped'])
        lost_packets = len(path_stats['lost'])
        received_packets = len(path_stats['received'])
        
        # 计算在途包数（仍在队列中的包）
        in_transit_packets = total_sent - (dropped_packets + lost_packets + received_packets)
        
        # 计算各种比率
        drop_rate = (dropped_packets / total_sent) * 100
        loss_rate = (lost_packets / total_sent) * 100
        success_rate = (received_packets / total_sent) * 100
        in_transit_rate = (in_transit_packets / total_sent) * 100
        
        print(f"带宽: {metrics['bandwidths'][-1]:.2f} Mbps")
        print(f"延迟: {metrics['delays'][-1]:.2f} ms")
        print(f"发送包数: {total_sent}")
        print(f"接收包数: {received_packets}")
        print(f"传输丢失包数: {lost_packets}")
        print(f"队列丢弃包数: {dropped_packets}")
        print(f"在途包数: {in_transit_packets}")
        print(f"队列丢弃率: {drop_rate:.2f}%")
        print(f"传输丢失率: {loss_rate:.2f}%")
        print(f"传输成功率: {success_rate:.2f}%")
        print(f"在途率: {in_transit_rate:.2f}%")
        
        # 验证统计的完整性
        total_percentage = drop_rate + loss_rate + success_rate + in_transit_rate
        if abs(total_percentage - 100) > 0.1:  # 允许0.1%的误差
            print(f"警告：数据包统计不匹配！总计: {total_percentage:.2f}%")
    
    print(f"路径长度: {len(path)}")
    print(f"奖励值: {metrics['rewards'][-1]:.2f}")
    print(f"探索率: {agent.epsilon:.3f}")
    print("-" * 50)

def train():
    # 初始化环境
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 初始化智能体
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 训练参数
    episodes = 2000
    batch_size = 32
    
    # 记录训练指标
    all_metrics = {
        'delays': [],
        'bandwidths': [],
        'losses': [],
        'path_lengths': [],
        'transmission_success_rates': [],
        'rewards': []
    }
    
    try:
        for episode in range(episodes):
            # 重置环境
            env.reset()
            
            # 选择不同MEO区域的源节点和目标节点
            leo_names = env.get_leo_names()
            meo_regions = set(env.leo_to_meo.values())
            source_meo = random.choice(list(meo_regions))
            dest_meo = random.choice([m for m in meo_regions if m != source_meo])
            
            # 从选定的MEO区域中选择源和目标LEO
            source_leos = [leo for leo, meo in env.leo_to_meo.items() if meo == source_meo]
            dest_leos = [leo for leo, meo in env.leo_to_meo.items() if meo == dest_meo]
            source = random.choice(source_leos)
            destination = random.choice(dest_leos)
            
            current_leo = source
            path = [current_leo]
            episode_reward = 0
            
            # 获取初始状态
            current_state = agent.get_state(env, current_leo, destination)
            
            # 重置统计数据
            path_stats = {
                'sent': set(),          # 所有生成的包
                'in_queue': set(),      # 在队列中的包
                'processed': set(),     # 已处理的包
                'dropped': set(),       # 因队列满被丢弃的包
                'lost': set(),          # 传输中丢失的包
                'received': set()       # 成功接收的包
            }
            
            while len(path) < MAX_PATH_LENGTH:
                # 获取可用动作
                available_actions = env.get_available_actions(current_leo)
                
                # 获取候选动作
                candidate_actions = env.get_candidate_actions(current_leo, destination, available_actions)
                
                # 选择动作
                action = agent.choose_action(current_state, candidate_actions or available_actions, 
                                          env, current_leo, destination, path)
                
                if action is None:
                    break
                
                # 执行动作
                next_state, reward, done, info = env.step(current_leo, action, path)
                
                if next_state is None:
                    break
                
                # 存储经验
                agent.memorize(current_state, action, reward, next_state, done)
                
                # 更新状态和路径
                current_state = next_state
                current_leo = info['next_leo']
                path.append(current_leo)
                episode_reward += reward
                
                # 经验回放
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                if done or current_leo == destination:
                    break
            
            # 更新目标网络
            if episode % 10 == 0:
                agent.update_target_network()
            
            # 记录指标
            path_stats = info['path_stats']
            total_sent = len(path_stats['sent'])
            if total_sent > 0:
                success_rate = (len(path_stats['received']) / total_sent) * 100
            else:
                success_rate = 0
            
            all_metrics['delays'].append(info['delay'])
            all_metrics['bandwidths'].append(info['bandwidth'])
            all_metrics['losses'].append(info['loss'])
            all_metrics['path_lengths'].append(len(path))
            all_metrics['transmission_success_rates'].append(success_rate)
            all_metrics['rewards'].append(episode_reward)
            
            # 打印统计信息
            print_episode_stats(episode, episodes, path, path_stats, all_metrics, agent, env)
            
            # 每100个episode保存一次指标图表
            if (episode + 1) % 100 == 0:
                plot_metrics(all_metrics)
                save_training_history(all_metrics)
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
    finally:
        # 保存最终的训练结果
        plot_metrics(all_metrics)
        save_training_history(all_metrics)
        print("\n训练完成")

if __name__ == '__main__':
    train() 