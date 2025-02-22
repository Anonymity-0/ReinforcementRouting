import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from datetime import datetime
from simulation.simulator import NetworkSimulator
from algorithms.ppo_algorithm import PPOAlgorithm
from algorithms.mappo_algorithm import MAPPOAlgorithm
from algorithms.dijkstra_algorithm import DijkstraAlgorithm

class RoutingBenchmark:
    """路由算法基准测试框架"""
    
    def __init__(self, env_config: Dict[str, Any], alg_config: Dict[str, Any]):
        """
        初始化路由基准测试
        
        Args:
            env_config: 环境配置
            alg_config: 算法配置
        """
        self.env_config = env_config
        self.alg_config = alg_config
        
        # 获取网络参数
        network_params = env_config['simulation']['network']
        
        # 创建算法配置
        ppo_config = {
            **alg_config['ppo'],
            'total_satellites': network_params['total_satellites'],
            'max_buffer_size': network_params['max_buffer_size'],
            'max_queue_length': network_params['max_queue_length'],
            'hidden_dim': alg_config['common']['hidden_dim']
        }
        
        mappo_config = {
            **alg_config['mappo'],
            'total_satellites': network_params['total_satellites'],
            'max_buffer_size': network_params['max_buffer_size'],
            'max_queue_length': network_params['max_queue_length'],
            'hidden_dim': alg_config['common']['hidden_dim']
        }
        
        # 初始化算法
        self.algorithms = {}
        self.algorithms['ppo'] = PPOAlgorithm(ppo_config)
        self.algorithms['mappo'] = MAPPOAlgorithm(mappo_config)
        self.algorithms['dijkstra'] = DijkstraAlgorithm(network_params)
        
        # 创建日志目录
        self.log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建结果目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = os.path.join('results', self.timestamp)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 创建仿真环境
        self.env = NetworkSimulator(self.env_config)
        
        # 训练和评估指标
        self.metrics = {
            'rewards': [],
            'delays': [],
            'packet_losses': [],
            'throughputs': [],
            'qos_satisfaction': []
        }
    
    def train(self, algorithm_name: str) -> None:
        """
        训练指定的算法
        
        Args:
            algorithm_name: 算法名称
        """
        print(f"开始训练 {algorithm_name} 算法...")
        
        # 获取算法实例
        algorithm = self.algorithms[algorithm_name]
        algorithm.train()  # 设置为训练模式
        
        # 训练参数
        max_episodes = self.alg_config['training']['max_episodes']
        max_steps = self.alg_config['training']['max_steps']
        save_interval = self.alg_config['training']['save_interval']
        
        # 创建保存目录
        save_dir = os.path.join(self.result_dir, algorithm_name, 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录训练指标
        episode_rewards = []
        episode_delays = []
        episode_losses = []
        episode_throughputs = []
        episode_qos = []
        
        for episode in range(max_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            
            # 收集每个episode的指标
            episode_delay = []
            episode_loss = []
            episode_throughput = []
            episode_qos_sat = []
            
            for step in range(max_steps):
                # 获取动作
                current_node = observation['packet']['current_node']
                target_node = observation['packet']['destination']
                action = algorithm.get_next_hop(current_node, target_node, observation)
                
                # 执行动作
                next_observation, reward, done, truncated, info = self.env.step(action)
                
                # 更新算法
                algorithm.update(
                    states=[observation],
                    actions=np.array([action]),
                    rewards=np.array([reward]),
                    next_states=[next_observation],
                    dones=np.array([done or truncated])
                )
                
                # 收集指标
                episode_reward += reward
                episode_delay.append(info['network_metrics']['delay'])
                episode_loss.append(info['network_metrics']['packet_loss'])
                episode_throughput.append(info['network_metrics']['throughput'])
                episode_qos_sat.append(np.mean(list(info['qos_satisfaction'].values())))
                
                if done or truncated:
                    break
                    
                observation = next_observation
            
            # 记录episode指标
            episode_rewards.append(episode_reward)
            episode_delays.append(np.mean(episode_delay))
            episode_losses.append(np.mean(episode_loss))
            episode_throughputs.append(np.mean(episode_throughput))
            episode_qos.append(np.mean(episode_qos_sat))
            
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{max_episodes}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Average Delay: {np.mean(episode_delay):.2f} ms")
                print(f"Average Loss: {np.mean(episode_loss):.4f}")
                print(f"Average Throughput: {np.mean(episode_throughput):.2f} Mbps")
                print(f"Average QoS Satisfaction: {np.mean(episode_qos_sat):.4f}\n")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                save_path = os.path.join(save_dir, f"model_episode_{episode + 1}.pt")
                algorithm.save(save_path)
                print(f"模型已保存到: {save_path}")
        
        # 保存训练指标
        self._save_metrics(
            algorithm_name=algorithm_name,
            mode='train',
            rewards=episode_rewards,
            delays=episode_delays,
            losses=episode_losses,
            throughputs=episode_throughputs,
            qos=episode_qos
        )
        
        # 绘制训练曲线
        self._plot_training_curves(
            algorithm_name=algorithm_name,
            rewards=episode_rewards,
            delays=episode_delays,
            losses=episode_losses,
            throughputs=episode_throughputs,
            qos=episode_qos
        )
    
    def evaluate(self, algorithm_name: str, model_path: str) -> Dict[str, float]:
        """
        评估指定的算法
        
        Args:
            algorithm_name: 算法名称
            model_path: 模型路径
            
        Returns:
            Dict[str, float]: 评估结果
        """
        print(f"开始评估 {algorithm_name} 算法...")
        
        # 获取算法实例并加载模型
        algorithm = self.algorithms[algorithm_name]
        algorithm.load(model_path)
        algorithm.eval()  # 设置为评估模式
        
        # 评估参数
        num_episodes = self.alg_config['evaluation']['eval_episodes']
        
        # 记录评估指标
        episode_rewards = []
        episode_delays = []
        episode_losses = []
        episode_throughputs = []
        episode_qos = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            
            # 收集每个episode的指标
            episode_delay = []
            episode_loss = []
            episode_throughput = []
            episode_qos_sat = []
            
            while True:
                # 获取动作
                current_node = observation['packet']['current_node']
                target_node = observation['packet']['destination']
                action = algorithm.get_next_hop(current_node, target_node, observation)
                
                # 执行动作
                next_observation, reward, done, truncated, info = self.env.step(action)
                
                # 收集指标
                episode_reward += reward
                episode_delay.append(info['network_metrics']['delay'])
                episode_loss.append(info['network_metrics']['packet_loss'])
                episode_throughput.append(info['network_metrics']['throughput'])
                episode_qos_sat.append(np.mean(list(info['qos_satisfaction'].values())))
                
                if done or truncated:
                    break
                    
                observation = next_observation
            
            # 记录episode指标
            episode_rewards.append(episode_reward)
            episode_delays.append(np.mean(episode_delay))
            episode_losses.append(np.mean(episode_loss))
            episode_throughputs.append(np.mean(episode_throughput))
            episode_qos.append(np.mean(episode_qos_sat))
            
            # 打印评估进度
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Average Delay: {np.mean(episode_delay):.2f} ms")
            print(f"Average Loss: {np.mean(episode_loss):.4f}")
            print(f"Average Throughput: {np.mean(episode_throughput):.2f} Mbps")
            print(f"Average QoS Satisfaction: {np.mean(episode_qos_sat):.4f}\n")
        
        # 保存评估指标
        self._save_metrics(
            algorithm_name=algorithm_name,
            mode='eval',
            rewards=episode_rewards,
            delays=episode_delays,
            losses=episode_losses,
            throughputs=episode_throughputs,
            qos=episode_qos
        )
        
        # 计算平均指标
        results = {
            'reward_mean': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards),
            'delay_mean': np.mean(episode_delays),
            'delay_std': np.std(episode_delays),
            'loss_mean': np.mean(episode_losses),
            'loss_std': np.std(episode_losses),
            'throughput_mean': np.mean(episode_throughputs),
            'throughput_std': np.std(episode_throughputs),
            'qos_mean': np.mean(episode_qos),
            'qos_std': np.std(episode_qos)
        }
        
        return results
    
    def compare_algorithms(self, model_paths: Dict[str, str]) -> None:
        """
        比较不同算法的性能
        
        Args:
            model_paths: 算法名称到模型路径的映射
        """
        print("开始算法对比实验...")
        
        # 收集每个算法的评估结果
        results = {}
        for algorithm_name, model_path in model_paths.items():
            results[algorithm_name] = self.evaluate(algorithm_name, model_path)
        
        # 创建对比图表
        self._plot_comparison(results)
        
        # 保存对比结果
        self._save_comparison(results)
    
    def _save_metrics(self,
                     algorithm_name: str,
                     mode: str,
                     rewards: List[float],
                     delays: List[float],
                     losses: List[float],
                     throughputs: List[float],
                     qos: List[float]) -> None:
        """
        保存指标数据
        
        Args:
            algorithm_name: 算法名称
            mode: 模式('train'或'eval')
            rewards: 奖励列表
            delays: 延迟列表
            losses: 丢包率列表
            throughputs: 吞吐量列表
            qos: QoS满意度列表
        """
        # 创建数据框
        df = pd.DataFrame({
            'reward': rewards,
            'delay': delays,
            'loss': losses,
            'throughput': throughputs,
            'qos': qos
        })
        
        # 保存到CSV文件
        save_path = os.path.join(self.result_dir, algorithm_name, f'{mode}_metrics.csv')
        df.to_csv(save_path, index=False)
        print(f"指标数据已保存到: {save_path}")
    
    def _plot_training_curves(self,
                            algorithm_name: str,
                            rewards: List[float],
                            delays: List[float],
                            losses: List[float],
                            throughputs: List[float],
                            qos: List[float]) -> None:
        """
        绘制训练曲线
        
        Args:
            algorithm_name: 算法名称
            rewards: 奖励列表
            delays: 延迟列表
            losses: 丢包率列表
            throughputs: 吞吐量列表
            qos: QoS满意度列表
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{algorithm_name} Training Curves')
        
        # 绘制奖励曲线
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 绘制延迟曲线
        axes[0, 1].plot(delays)
        axes[0, 1].set_title('Delay')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Delay (ms)')
        
        # 绘制丢包率曲线
        axes[1, 0].plot(losses)
        axes[1, 0].set_title('Packet Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss Rate')
        
        # 绘制吞吐量曲线
        axes[1, 1].plot(throughputs)
        axes[1, 1].set_title('Throughput')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Throughput (Mbps)')
        
        # 绘制QoS满意度曲线
        axes[2, 0].plot(qos)
        axes[2, 0].set_title('QoS Satisfaction')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Satisfaction Rate')
        
        # 保存图表
        save_path = os.path.join(self.result_dir, algorithm_name, 'training_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"训练曲线已保存到: {save_path}")
    
    def _plot_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        绘制算法对比图表
        
        Args:
            results: 算法评估结果
        """
        metrics = ['reward', 'delay', 'loss', 'throughput', 'qos']
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Comparison')
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            # 提取数据
            means = [results[alg][f'{metric}_mean'] for alg in results]
            stds = [results[alg][f'{metric}_std'] for alg in results]
            
            # 绘制柱状图
            x = np.arange(len(results))
            axes[row, col].bar(x, means, yerr=stds, capsize=5)
            axes[row, col].set_title(metric.capitalize())
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(list(results.keys()))
        
        # 保存图表
        save_path = os.path.join(self.result_dir, 'algorithm_comparison.png')
        plt.savefig(save_path)
        plt.close()
        print(f"对比图表已保存到: {save_path}")
    
    def _save_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        保存对比结果
        
        Args:
            results: 算法评估结果
        """
        # 创建数据框
        df = pd.DataFrame(results).T
        
        # 保存到CSV文件
        save_path = os.path.join(self.result_dir, 'comparison_results.csv')
        df.to_csv(save_path)
        print(f"对比结果已保存到: {save_path}") 