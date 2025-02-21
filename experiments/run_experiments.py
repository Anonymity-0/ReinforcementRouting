import os
import torch
import numpy as np
from datetime import datetime
from simulation.simulator import NetworkSimulator
from algorithms.ppo_algorithm import PPOAlgorithm
import yaml
import logging
from logging import Logger
from algorithms.dijkstra_algorithm import DijkstraAlgorithm
from tqdm import tqdm
from algorithms.mappo_algorithm import MAPPOAlgorithm

def setup_logging(save_dir):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置文件日志
    log_file = os.path.join(save_dir, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_ppo(env: NetworkSimulator, save_dir: str, logger: Logger):
    """
    训练PPO算法
    
    Args:
        env: 网络仿真环境
        save_dir: 模型保存目录
        logger: 日志记录器
    """
    # 获取PPO配置
    ppo_config = {
        'total_satellites': env.total_satellites,
        'max_buffer_size': env.config['simulation']['network']['max_buffer_size'],
        'max_queue_length': env.config['simulation']['network']['max_queue_length'],
        'device': env.config['algorithm']['common']['device'],
        'hidden_dim': env.config['algorithm']['common']['hidden_dim'],
        'learning_rate': env.config['algorithm']['ppo']['learning_rate'],
        'gamma': env.config['algorithm']['ppo']['gamma'],
        'gae_lambda': env.config['algorithm']['ppo']['gae_lambda'],
        'clip_param': env.config['algorithm']['ppo']['clip_param'],
        'num_epochs': env.config['algorithm']['ppo']['num_epochs'],
        'batch_size': env.config['algorithm']['ppo']['batch_size'],
        'value_loss_coef': env.config['algorithm']['ppo']['value_loss_coef'],
        'entropy_coef': env.config['algorithm']['ppo']['entropy_coef'],
        'max_grad_norm': env.config['algorithm']['ppo']['max_grad_norm'],
        'buffer_size': env.config['algorithm']['ppo']['buffer_size'],
        'initial_epsilon': env.config['algorithm']['ppo']['initial_epsilon'],
        'final_epsilon': env.config['algorithm']['ppo']['final_epsilon'],
        'initial_temperature': env.config['algorithm']['ppo']['initial_temperature'],
        'final_temperature': env.config['algorithm']['ppo']['final_temperature']
    }
    
    # 初始化PPO算法
    algorithm = PPOAlgorithm(ppo_config)
    env.set_routing_algorithm(algorithm)
    
    # 设置训练参数
    max_episodes = env.config['simulation']['common']['max_episodes']
    max_steps = env.config['simulation']['common']['max_steps']
    
    # 记录延迟统计
    delays = []
    
    # 开始训练
    logger.info(f"开始PPO训练 - 设备: {ppo_config['device']}")
    
    # 创建episode进度条
    episode_pbar = tqdm(range(max_episodes), desc="训练进度", unit="episode")
    
    # 训练循环
    for episode in episode_pbar:
        state = env.reset()
        episode_reward = 0
        
        # 每个episode的统计
        episode_packets = {
            'generated': 0,
            'delivered': 0,
            'dropped': 0,
            'in_transit': 0
        }
        
        # 记录丢包原因
        drop_reasons = {
            'no_valid_path': 0,  # 找不到有效路径
            'buffer_overflow': 0,  # 缓冲区溢出
            'link_failure': 0,    # 链路故障
            'other': 0            # 其他原因
        }
        
        # 创建step进度条
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode+1}", leave=False, unit="step")
        
        # 收集一个episode的数据
        for step in step_pbar:
            current_node = state['packet']['current_node']
            target_node = state['packet']['destination']
            action = algorithm.get_next_hop(current_node, target_node, state)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            algorithm.update([state], [action], [reward], [next_state], [done])
            
            # 更新统计信息
            if 'metrics' in info:
                metrics = info['metrics']
                
                # 更新生成的数据包数量
                if 'packets_generated' in metrics:
                    new_packets = max(0, metrics['packets_generated'] - episode_packets['generated'])
                    if new_packets > 0:
                        episode_packets['generated'] += new_packets
                        episode_packets['in_transit'] += new_packets
                
                # 更新成功传输的数据包数量
                if 'packets_delivered' in metrics:
                    new_delivered = max(0, metrics['packets_delivered'] - episode_packets['delivered'])
                    if new_delivered > 0:
                        episode_packets['delivered'] += new_delivered
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_delivered)
                        
                        # 记录延迟
                        if 'delay' in metrics and metrics['delay'] > 0:
                            delays.append(metrics['delay'])
                
                # 更新丢弃的数据包数量
                if 'packets_dropped' in metrics:
                    new_dropped = max(0, metrics['packets_dropped'] - episode_packets['dropped'])
                    if new_dropped > 0:
                        episode_packets['dropped'] += new_dropped
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_dropped)
                        
                        # 记录丢包原因
                        if action == -1:
                            drop_reasons['no_valid_path'] += new_dropped
                        if metrics.get('buffer_overflow', False):
                            drop_reasons['buffer_overflow'] += new_dropped
                        if metrics.get('link_failure', False):
                            drop_reasons['link_failure'] += new_dropped
                        if not (action == -1 or metrics.get('buffer_overflow', False) or metrics.get('link_failure', False)):
                            drop_reasons['other'] += new_dropped
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            # 更新进度条描述
            step_pbar.set_postfix({
                '奖励': f"{episode_reward:.2f}",
                '成功率': f"{(episode_packets['delivered']/max(1, episode_packets['generated'])*100):.2f}%"
            })
            
            if done:
                break
        
        # 关闭step进度条
        step_pbar.close()
        
        # 计算详细统计信息
        total_packets = episode_packets['generated']
        delivered_packets = episode_packets['delivered']
        dropped_packets = episode_packets['dropped']
        in_transit_packets = episode_packets['in_transit']
        
        # 计算各种比率
        delivery_rate = (delivered_packets / max(1, total_packets)) * 100
        drop_rate = (dropped_packets / max(1, total_packets)) * 100
        
        # 计算延迟统计
        if delays:
            avg_delay = np.mean(delays)
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            std_delay = np.std(delays)
        else:
            avg_delay = min_delay = max_delay = std_delay = 0.0
        
        # 计算丢包原因分布
        drop_reason_stats = {}
        if dropped_packets > 0:
            for reason, count in drop_reasons.items():
                drop_reason_stats[reason] = (count / dropped_packets) * 100
        
        # 记录详细日志
        if (episode + 1) % 10 == 0:  # 每10个episode记录一次详细日志
            logger.info(f"\nEpisode {episode + 1} 详细统计:")
            logger.info(f"数据包统计:")
            logger.info(f"  - 总数据包: {total_packets}")
            logger.info(f"  - 成功传输: {delivered_packets} ({delivery_rate:.2f}%)")
            logger.info(f"  - 丢弃数据包: {dropped_packets} ({drop_rate:.2f}%)")
            logger.info(f"  - 传输中: {in_transit_packets}")
            
            logger.info(f"\n延迟统计:")
            logger.info(f"  - 平均延迟: {avg_delay:.3f}秒")
            logger.info(f"  - 最小延迟: {min_delay:.3f}秒")
            logger.info(f"  - 最大延迟: {max_delay:.3f}秒")
            logger.info(f"  - 延迟标准差: {std_delay:.3f}秒")
            
            logger.info(f"\n丢包原因分析:")
            for reason, percentage in drop_reason_stats.items():
                logger.info(f"  - {reason}: {drop_reasons[reason]}个 ({percentage:.2f}%)")
        
        # 衰减探索率
        if algorithm.epsilon > algorithm.config.get('final_epsilon', 0.01):
            algorithm.epsilon -= (algorithm.config.get('initial_epsilon', 0.3) - algorithm.config.get('final_epsilon', 0.01)) / (max_episodes * 0.8)
        
        # 衰减温度参数
        if algorithm.policy_net.temperature.item() > algorithm.config.get('final_temperature', 0.5):
            algorithm.policy_net.temperature.data -= torch.ones(1).to(algorithm.device) * (
                (algorithm.config.get('initial_temperature', 1.0) - algorithm.config.get('final_temperature', 0.5)) / (max_episodes * 0.8)
            )
        
        # 更新episode进度条描述
        episode_pbar.set_postfix({
            '奖励': f"{episode_reward/max_steps:.2f}",
            '成功率': f"{delivery_rate:.1f}%",
            '丢包率': f"{drop_rate:.1f}%",
            '延迟': f"{avg_delay:.2f}s",
            'ε': f"{algorithm.epsilon:.2f}"
        })
        
        # 每100个episode保存一次模型
        if (episode + 1) % 100 == 0:
            save_path = os.path.join(save_dir, f'ppo_model_episode_{episode+1}.pth')
            algorithm.save(save_path)
            logger.info(f"模型已保存到: {save_path}")
    
    # 关闭episode进度条
    episode_pbar.close()
    
    # 保存最终模型
    final_save_path = os.path.join(save_dir, 'ppo_model_final.pth')
    algorithm.save(final_save_path)
    logger.info(f"最终模型已保存到: {final_save_path}")
    
    return algorithm

def evaluate_ppo(env, model_path, logger, num_episodes=100):
    """评估PPO算法"""
    # 加载模型
    device = torch.device(env.algorithm_config['common'].get('device', 'cpu'))
    algorithm = PPOAlgorithm(
        state_dim=env.observation_space['topology']['positions'].shape[0] * 3 +
                 env.observation_space['topology']['velocities'].shape[0] * 3 +
                 env.observation_space['network']['queue_lengths'].shape[0] +
                 env.observation_space['network']['link_states'].shape[0] * env.observation_space['network']['link_states'].shape[1],
        action_dim=env.action_space.n,
        hidden_dim=env.algorithm_config['common'].get('hidden_dim', 256),
        device=device
    )
    algorithm.load(model_path)
    
    logger.info(f"开始评估PPO模型: {model_path}")
    
    total_rewards = []
    metrics = {
        'delay': [],
        'throughput': [],
        'packet_loss': [],
        'link_utilization': []
    }
    
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            state = env.get_state()
            action = algorithm.select_action(state, deterministic=True)  # 评估时使用确定性策略
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # 收集指标
            for key in metrics:
                if key in info['metrics']:
                    metrics[key].extend(info['metrics'][key])
            
            if done:
                break
                
        total_rewards.append(episode_reward)
        logger.info(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    # 计算平均指标
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_delay': np.mean(metrics['delay']) if metrics['delay'] else 0,
        'mean_throughput': np.mean(metrics['throughput']) if metrics['throughput'] else 0,
        'mean_packet_loss': np.mean(metrics['packet_loss']) if metrics['packet_loss'] else 0,
        'mean_link_utilization': np.mean(metrics['link_utilization']) if metrics['link_utilization'] else 0
    }
    
    logger.info("\n评估结果:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
        
    return results 

def run_dijkstra(env: NetworkSimulator, save_dir: str, logger: Logger):
    """
    运行Dijkstra算法
    
    Args:
        env: 网络仿真环境
        save_dir: 结果保存目录
        logger: 日志记录器
    """
    # 初始化Dijkstra算法
    dijkstra_config = {
        'total_satellites': env.total_satellites
    }
    
    algorithm = DijkstraAlgorithm(dijkstra_config)
    env.set_routing_algorithm(algorithm)
    
    # 重置环境和指标
    state = env.reset()
    
    # 使用配置文件中的步数
    max_episodes = env.config['simulation']['common']['max_episodes']
    max_steps = env.config['simulation']['common']['max_steps']
    
    # 记录延迟统计
    delays = []
    
    # 运行多个episode
    for episode in range(max_episodes):
        logger.info(f"\n开始Episode {episode + 1}/{max_episodes}")
        state = env.reset()
        
        # 每个episode的统计
        episode_packets = {
            'generated': 0,
            'delivered': 0,
            'dropped': 0,
            'in_transit': 0
        }
        
        # 记录丢包原因
        drop_reasons = {
            'no_valid_path': 0,  # 找不到有效路径
            'buffer_overflow': 0,  # 缓冲区溢出
            'link_failure': 0,    # 链路故障
            'other': 0            # 其他原因
        }
        
        for step in range(max_steps):
            # 获取当前节点和目标节点
            current_node = state['packet']['current_node']
            target_node = state['packet']['destination']
            
            # 使用Dijkstra算法计算下一跳
            next_hop = algorithm.get_next_hop(current_node, target_node, state)
            
            # 执行动作并获取新状态
            next_state, reward, done, info = env.step(next_hop)
            
            # 更新统计信息
            if 'metrics' in info:
                metrics = info['metrics']
                
                # 更新生成的数据包数量
                if 'packets_generated' in metrics:
                    new_packets = max(0, metrics['packets_generated'] - episode_packets['generated'])
                    if new_packets > 0:
                        episode_packets['generated'] += new_packets
                        episode_packets['in_transit'] += new_packets
                
                # 更新成功传输的数据包数量
                if 'packets_delivered' in metrics:
                    new_delivered = max(0, metrics['packets_delivered'] - episode_packets['delivered'])
                    if new_delivered > 0:
                        episode_packets['delivered'] += new_delivered
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_delivered)
                        
                        # 记录延迟
                        if 'delay' in metrics and metrics['delay'] > 0:
                            delays.append(metrics['delay'])
                
                # 更新丢弃的数据包数量
                if 'packets_dropped' in metrics:
                    new_dropped = max(0, metrics['packets_dropped'] - episode_packets['dropped'])
                    if new_dropped > 0:
                        episode_packets['dropped'] += new_dropped
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_dropped)
                        
                        # 记录丢包原因（一个包可能有多个丢弃原因）
                        if next_hop == -1:
                            drop_reasons['no_valid_path'] += new_dropped
                        if metrics.get('buffer_overflow', False):
                            drop_reasons['buffer_overflow'] += new_dropped
                        if metrics.get('link_failure', False):
                            drop_reasons['link_failure'] += new_dropped
                        if not (next_hop == -1 or metrics.get('buffer_overflow', False) or metrics.get('link_failure', False)):
                            drop_reasons['other'] += new_dropped
            
            # 更新状态
            state = next_state
            
            # 如果到达目标节点或无法继续，开始新的数据包
            if done:
                state = env.reset()
        
        # 计算延迟统计
        if delays:
            avg_delay = np.mean(delays)
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            std_delay = np.std(delays)
        else:
            avg_delay = min_delay = max_delay = std_delay = 0.0
        
        # 打印最终统计结果
        logger.info("\nDijkstra算法运行统计:")
        logger.info(f"总步数: {max_steps}")
        logger.info(f"生成的数据包总数: {episode_packets['generated']}")
        logger.info(f"成功传输的数据包数: {episode_packets['delivered']}")
        logger.info(f"丢弃的数据包数: {episode_packets['dropped']}")
        logger.info(f"传输中的数据包数: {episode_packets['in_transit']}")
        
        if episode_packets['generated'] > 0:
            success_rate = (episode_packets['delivered'] / episode_packets['generated']) * 100
            logger.info(f"传输成功率: {success_rate:.2f}%")
        
        # 打印延迟统计
        logger.info("\n延迟统计:")
        logger.info(f"平均延迟: {avg_delay:.3f}秒")
        logger.info(f"最小延迟: {min_delay:.3f}秒")
        logger.info(f"最大延迟: {max_delay:.3f}秒")
        logger.info(f"延迟标准差: {std_delay:.3f}秒")
        
        # 打印丢包原因分析
        logger.info("\n丢包原因分析:")
        total_drops = episode_packets['dropped']  # 使用实际丢弃的数据包总数
        if total_drops > 0:
            for reason, count in drop_reasons.items():
                percentage = (count / total_drops * 100)
                logger.info(f"{reason}: {count}个 ({percentage:.2f}%)")
        else:
            for reason in drop_reasons:
                logger.info(f"{reason}: 0个 (0.00%)")
    
    return algorithm 

def train_mappo(env: NetworkSimulator, save_dir: str, logger: Logger):
    """
    训练MAPPO算法
    
    Args:
        env: 网络仿真环境
        save_dir: 模型保存目录
        logger: 日志记录器
    """
    # 获取MAPPO配置
    mappo_config = {
        'total_satellites': env.total_satellites,
        'max_buffer_size': env.config['simulation']['network']['max_buffer_size'],
        'max_queue_length': env.config['simulation']['network']['max_queue_length'],
        'device': env.config['algorithm']['common']['device'],
        'hidden_dim': env.config['algorithm']['common']['hidden_dim'],
        'learning_rate': env.config['algorithm']['mappo']['learning_rate'],
        'gamma': env.config['algorithm']['mappo']['gamma'],
        'gae_lambda': env.config['algorithm']['mappo']['gae_lambda'],
        'clip_param': env.config['algorithm']['mappo']['clip_param'],
        'num_epochs': env.config['algorithm']['mappo']['num_epochs'],
        'batch_size': env.config['algorithm']['mappo']['batch_size'],
        'value_loss_coef': env.config['algorithm']['mappo']['value_loss_coef'],
        'entropy_coef': env.config['algorithm']['mappo']['entropy_coef'],
        'max_grad_norm': env.config['algorithm']['mappo']['max_grad_norm'],
        'use_centralized_critic': env.config['algorithm']['mappo']['use_centralized_critic'],
        'use_reward_normalization': env.config['algorithm']['mappo']['use_reward_normalization'],
        'use_advantage_normalization': env.config['algorithm']['mappo']['use_advantage_normalization']
    }
    
    # 初始化MAPPO算法
    algorithm = MAPPOAlgorithm(mappo_config)
    env.set_routing_algorithm(algorithm)
    
    # 设置训练参数
    max_episodes = env.config['simulation']['common']['max_episodes']
    max_steps = env.config['simulation']['common']['max_steps']
    
    # 记录延迟统计
    delays = []
    
    # 开始训练
    logger.info(f"开始MAPPO训练 - 设备: {mappo_config['device']}")
    
    # 创建episode进度条
    episode_pbar = tqdm(range(max_episodes), desc="训练进度", unit="episode")
    
    # 训练循环
    for episode in episode_pbar:
        state = env.reset()
        episode_reward = 0
        
        # 每个episode的统计
        episode_packets = {
            'generated': 0,
            'delivered': 0,
            'dropped': 0,
            'in_transit': 0
        }
        
        # 记录丢包原因
        drop_reasons = {
            'no_valid_path': 0,  # 找不到有效路径
            'buffer_overflow': 0,  # 缓冲区溢出
            'link_failure': 0,    # 链路故障
            'other': 0            # 其他原因
        }
        
        # 创建step进度条
        step_pbar = tqdm(range(max_steps), desc=f"Episode {episode+1}", leave=False, unit="step")
        
        # 收集一个episode的数据
        for step in step_pbar:
            current_node = state['packet']['current_node']
            target_node = state['packet']['destination']
            action = algorithm.get_next_hop(current_node, target_node, state)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            algorithm.update([state], [action], [reward], [next_state], [done])
            
            # 更新统计信息
            if 'metrics' in info:
                metrics = info['metrics']
                
                # 更新生成的数据包数量
                if 'packets_generated' in metrics:
                    new_packets = max(0, metrics['packets_generated'] - episode_packets['generated'])
                    if new_packets > 0:
                        episode_packets['generated'] += new_packets
                        episode_packets['in_transit'] += new_packets
                
                # 更新成功传输的数据包数量
                if 'packets_delivered' in metrics:
                    new_delivered = max(0, metrics['packets_delivered'] - episode_packets['delivered'])
                    if new_delivered > 0:
                        episode_packets['delivered'] += new_delivered
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_delivered)
                        
                        # 记录延迟
                        if 'delay' in metrics and metrics['delay'] > 0:
                            delays.append(metrics['delay'])
                
                # 更新丢弃的数据包数量
                if 'packets_dropped' in metrics:
                    new_dropped = max(0, metrics['packets_dropped'] - episode_packets['dropped'])
                    if new_dropped > 0:
                        episode_packets['dropped'] += new_dropped
                        episode_packets['in_transit'] = max(0, episode_packets['in_transit'] - new_dropped)
                        
                        # 记录丢包原因
                        if action == -1:
                            drop_reasons['no_valid_path'] += new_dropped
                        if metrics.get('buffer_overflow', False):
                            drop_reasons['buffer_overflow'] += new_dropped
                        if metrics.get('link_failure', False):
                            drop_reasons['link_failure'] += new_dropped
                        if not (action == -1 or metrics.get('buffer_overflow', False) or metrics.get('link_failure', False)):
                            drop_reasons['other'] += new_dropped
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            # 更新进度条描述
            step_pbar.set_postfix({
                '奖励': f"{episode_reward:.2f}",
                '成功率': f"{(episode_packets['delivered']/max(1, episode_packets['generated'])*100):.2f}%"
            })
            
            if done:
                break
        
        # 关闭step进度条
        step_pbar.close()
        
        # 计算详细统计信息
        total_packets = episode_packets['generated']
        delivered_packets = episode_packets['delivered']
        dropped_packets = episode_packets['dropped']
        in_transit_packets = episode_packets['in_transit']
        
        # 计算各种比率
        delivery_rate = (delivered_packets / max(1, total_packets)) * 100
        drop_rate = (dropped_packets / max(1, total_packets)) * 100
        
        # 计算延迟统计
        if delays:
            avg_delay = np.mean(delays)
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            std_delay = np.std(delays)
        else:
            avg_delay = min_delay = max_delay = std_delay = 0.0
        
        # 计算丢包原因分布
        drop_reason_stats = {}
        if dropped_packets > 0:
            for reason, count in drop_reasons.items():
                drop_reason_stats[reason] = (count / dropped_packets) * 100
        
        # 记录详细日志
        if (episode + 1) % 10 == 0:  # 每10个episode记录一次详细日志
            logger.info(f"\nEpisode {episode + 1} 详细统计:")
            logger.info(f"数据包统计:")
            logger.info(f"  - 总数据包: {total_packets}")
            logger.info(f"  - 成功传输: {delivered_packets} ({delivery_rate:.2f}%)")
            logger.info(f"  - 丢弃数据包: {dropped_packets} ({drop_rate:.2f}%)")
            logger.info(f"  - 传输中: {in_transit_packets}")
            
            logger.info(f"\n延迟统计:")
            logger.info(f"  - 平均延迟: {avg_delay:.3f}秒")
            logger.info(f"  - 最小延迟: {min_delay:.3f}秒")
            logger.info(f"  - 最大延迟: {max_delay:.3f}秒")
            logger.info(f"  - 延迟标准差: {std_delay:.3f}秒")
            
            logger.info(f"\n丢包原因分析:")
            for reason, percentage in drop_reason_stats.items():
                logger.info(f"  - {reason}: {drop_reasons[reason]}个 ({percentage:.2f}%)")
        
        # 更新episode进度条描述
        episode_pbar.set_postfix({
            '奖励': f"{episode_reward/max_steps:.2f}",
            '成功率': f"{delivery_rate:.1f}%",
            '丢包率': f"{drop_rate:.1f}%",
            '延迟': f"{avg_delay:.2f}s"
        })
        
        # 每100个episode保存一次模型
        if (episode + 1) % 100 == 0:
            save_path = os.path.join(save_dir, f'mappo_model_episode_{episode+1}.pth')
            algorithm.save(save_path)
            logger.info(f"模型已保存到: {save_path}")
    
    # 关闭episode进度条
    episode_pbar.close()
    
    # 保存最终模型
    final_save_path = os.path.join(save_dir, 'mappo_model_final.pth')
    algorithm.save(final_save_path)
    logger.info(f"最终模型已保存到: {final_save_path}")
    
    return algorithm 