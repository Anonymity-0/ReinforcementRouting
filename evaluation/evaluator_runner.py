import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

from .evaluation import Evaluator
from visualization.visualizer import Visualizer
from environment.network_environment import NetworkEnvironment

class EvaluatorRunner:
    """评估运行器类"""
    
    def __init__(self,
                config: Dict,
                env: NetworkEnvironment,
                algorithm,
                evaluator: Evaluator,
                visualizer: Visualizer,
                logger: logging.Logger,
                save_dir: str):
        """
        初始化评估运行器
        
        Args:
            config: 配置字典
            env: 环境实例
            algorithm: 算法实例
            evaluator: 评估器实例
            visualizer: 可视化器实例
            logger: 日志记录器
            save_dir: 保存目录
        """
        self.config = config
        self.env = env
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.visualizer = visualizer
        self.logger = logger
        self.save_dir = save_dir
    
    def collect_metrics(self,
                      sat_id: int,
                      qos_type: str) -> Dict:
        """
        收集性能指标
        
        Args:
            sat_id: 卫星ID
            qos_type: QoS类型
            
        Returns:
            Dict: 性能指标
        """
        state = self.env.get_state(sat_id)
        neighbors = self.env.get_neighbors(sat_id)
        metrics = {}
        
        for neighbor in neighbors:
            edge = self.env.network.edges[sat_id, neighbor]
            
            # 计算延迟并确保非负
            queuing_delay = self.env._calculate_queuing_delay(
                self.env.network.nodes[neighbor]['queue_length'],
                edge['capacity'] / self.env.packet_size
            )
            delay = max(0.0, (edge['propagation_delay'] + queuing_delay) * 1000)  # 转换为毫秒
            
            # 计算丢包率并限制在[0,1]范围内
            packet_loss = self.env.network.nodes[neighbor]['queue_length'] / self.env.buffer_size
            packet_loss = max(0.0, min(1.0, packet_loss))
            
            # 计算吞吐量并确保非负
            throughput = max(0.0, edge['capacity'] / 1e6)  # 转换为Mbps
            
            metrics[neighbor] = {
                'delay': delay,
                'packet_loss': packet_loss,
                'throughput': throughput
            }
        
        return metrics
    
    def run_evaluation(self) -> Dict:
        """
        运行评估
        
        Returns:
            Dict: 评估结果
        """
        self.logger.info("开始评估...")
        
        # 获取评估参数
        eval_episodes = self.config['evaluation']['eval_episodes']
        
        # 评估结果
        results = {
            'episode_rewards': [],
            'success_rate': [],
            'average_delay': [],
            'average_packet_loss': [],
            'average_throughput': [],
            'qos_satisfaction': []
        }
        
        # 评估循环
        for episode in range(eval_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            success_count = 0
            step_count = 0
            
            # 收集指标
            metrics = {
                'delays': {'total': [], 'delay_sensitive': [],
                          'reliability_sensitive': [], 'throughput_sensitive': []},
                'packet_losses': {'total': [], 'delay_sensitive': [],
                                'reliability_sensitive': [], 'throughput_sensitive': []},
                'throughputs': {'total': [], 'delay_sensitive': [],
                              'reliability_sensitive': [], 'throughput_sensitive': []}
            }
            
            done = False
            while not done:
                # 选择动作
                action = self.algorithm.act(state, deterministic=True)
                
                # 执行动作
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 收集指标
                for qos_type in ['delay_sensitive', 'reliability_sensitive',
                               'throughput_sensitive']:
                    sat_metrics = self.collect_metrics(action, qos_type)
                    
                    for neighbor, values in sat_metrics.items():
                        metrics['delays'][qos_type].append(values['delay'])
                        metrics['packet_losses'][qos_type].append(values['packet_loss'])
                        metrics['throughputs'][qos_type].append(values['throughput'])
                        
                        metrics['delays']['total'].append(values['delay'])
                        metrics['packet_losses']['total'].append(values['packet_loss'])
                        metrics['throughputs']['total'].append(values['throughput'])
                
                # 检查是否成功
                if reward > 0:
                    success_count += 1
                
                state = next_state
            
            # 更新评估器
            self.evaluator.update(
                metrics['delays'],
                metrics['packet_losses'],
                metrics['throughputs']
            )
            
            # 记录结果
            results['episode_rewards'].append(episode_reward)
            results['success_rate'].append(success_count / step_count)
            results['average_delay'].append(np.mean(metrics['delays']['total']))
            results['average_packet_loss'].append(np.mean(metrics['packet_losses']['total']))
            results['average_throughput'].append(np.mean(metrics['throughputs']['total']))
            
            # 获取QoS满意度
            qos_metrics = self.evaluator.get_latest_metrics()['qos_satisfaction']
            avg_satisfaction = np.mean(list(qos_metrics.values()))
            results['qos_satisfaction'].append(avg_satisfaction)
            
            # 记录日志
            self.logger.info(f"Episode {episode + 1}/{eval_episodes}:")
            self.logger.info(f"  Reward: {episode_reward:.4f}")
            self.logger.info(f"  Success Rate: {success_count / step_count:.4f}")
            self.logger.info(f"  Average Delay: {results['average_delay'][-1]:.4f} ms")
            self.logger.info(f"  Average Packet Loss: {results['average_packet_loss'][-1]:.4f}")
            self.logger.info(f"  Average Throughput: {results['average_throughput'][-1]:.4f} Mbps")
            self.logger.info(f"  QoS Satisfaction: {avg_satisfaction:.4f}")
        
        # 计算总体统计信息
        final_results = {
            'mean_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'mean_success_rate': np.mean(results['success_rate']),
            'mean_delay': np.mean(results['average_delay']),
            'mean_packet_loss': np.mean(results['average_packet_loss']),
            'mean_throughput': np.mean(results['average_throughput']),
            'mean_qos_satisfaction': np.mean(results['qos_satisfaction'])
        }
        
        # 保存评估结果
        self.save_results(results, final_results)
        
        # 绘制评估结果
        self.plot_results()
        
        self.logger.info("评估完成!")
        self.logger.info("最终结果:")
        for key, value in final_results.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return final_results
    
    def save_results(self, results: Dict, final_results: Dict) -> None:
        """
        保存评估结果
        
        Args:
            results: 每轮评估结果
            final_results: 最终统计结果
        """
        results_file = os.path.join(self.save_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'episode_results': results,
                'final_results': final_results
            }, f, indent=4)
    
    def plot_results(self) -> None:
        """绘制评估结果"""
        if self.config['visualization']['plot_metrics']:
            self.visualizer.plot_performance_metrics(self.evaluator.metrics, 'delay')
            self.visualizer.plot_performance_metrics(self.evaluator.metrics, 'packet_loss')
            self.visualizer.plot_performance_metrics(self.evaluator.metrics, 'throughput')
            self.visualizer.plot_qos_comparison(self.evaluator.metrics) 
 