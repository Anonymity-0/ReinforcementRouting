import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json

class ResultPlotter:
    """结果可视化类"""
    
    def __init__(self, save_dir: str):
        """
        初始化结果可视化器
        
        Args:
            save_dir: 结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_training_curves(self,
                           rewards: List[float],
                           success_rates: List[float],
                           delays: List[float],
                           packet_losses: List[float],
                           throughputs: List[float],
                           algorithm_name: str) -> None:
        """
        绘制训练曲线
        
        Args:
            rewards: 奖励列表
            success_rates: 成功率列表
            delays: 延迟列表
            packet_losses: 丢包率列表
            throughputs: 吞吐量列表
            algorithm_name: 算法名称
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{algorithm_name} 训练曲线')
        
        # 绘制奖励曲线
        axes[0, 0].plot(rewards, 'b-', label='奖励')
        axes[0, 0].set_title('奖励值')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # 绘制成功率曲线
        axes[0, 1].plot(success_rates, 'g-', label='成功率')
        axes[0, 1].set_title('传输成功率')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('成功率 (%)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 绘制延迟曲线
        axes[1, 0].plot(delays, 'r-', label='延迟')
        axes[1, 0].set_title('平均延迟')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('延迟 (ms)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # 绘制丢包率曲线
        axes[1, 1].plot(packet_losses, 'm-', label='丢包率')
        axes[1, 1].set_title('丢包率')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_ylabel('丢包率 (%)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # 绘制吞吐量曲线
        axes[2, 0].plot(throughputs, 'c-', label='吞吐量')
        axes[2, 0].set_title('平均吞吐量')
        axes[2, 0].set_xlabel('训练轮次')
        axes[2, 0].set_ylabel('吞吐量 (Mbps)')
        axes[2, 0].grid(True)
        axes[2, 0].legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, f'{algorithm_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_evaluation_results(self, results: Dict[str, float], algorithm_name: str) -> None:
        """
        绘制评估结果
        
        Args:
            results: 评估结果字典
            algorithm_name: 算法名称
        """
        # 提取指标
        metrics = {
            '平均奖励': results['mean_reward'],
            '奖励标准差': results['std_reward'],
            '平均延迟': results['mean_delay'],
            '平均吞吐量': results['mean_throughput'],
            '平均丢包率': results['mean_packet_loss'],
            '平均链路利用率': results['mean_link_utilization']
        }
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        bars = ax.bar(x, list(metrics.values()))
        
        # 设置标题和标签
        ax.set_title(f'{algorithm_name} 评估结果')
        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        
        # 在柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, f'{algorithm_name}_evaluation_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_algorithm_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        绘制算法对比图
        
        Args:
            results: 不同算法的评估结果
        """
        # 提取要比较的指标
        metrics = ['mean_reward', 'mean_delay', 'mean_throughput', 'mean_packet_loss', 'mean_link_utilization']
        metric_names = ['平均奖励', '平均延迟', '平均吞吐量', '平均丢包率', '平均链路利用率']
        
        # 创建子图
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('算法性能对比')
        
        # 扁平化axes数组以便迭代
        axes_flat = axes.flatten()
        
        # 为每个指标创建柱状图
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                # 提取数据
                alg_names = list(results.keys())
                values = [results[alg][metric] for alg in alg_names]
                
                # 创建柱状图
                bars = ax.bar(range(len(alg_names)), values)
                
                # 设置标题和标签
                ax.set_title(metric_name)
                ax.set_xticks(range(len(alg_names)))
                ax.set_xticklabels(alg_names, rotation=45, ha='right')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom')
                
                ax.grid(True)
        
        # 移除多余的子图
        for i in range(len(metrics), len(axes_flat)):
            fig.delaxes(axes_flat[i])
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, 'algorithm_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, results: Dict[str, float], algorithm_name: str) -> None:
        """
        保存评估结果到JSON文件
        
        Args:
            results: 评估结果
            algorithm_name: 算法名称
        """
        save_path = os.path.join(self.save_dir, f'{algorithm_name}_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4) 