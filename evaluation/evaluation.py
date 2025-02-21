import numpy as np
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

class Evaluator:
    """评估器类"""
    
    def __init__(self, config: Dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化指标存储
        self.metrics = {
            'delay': {
                'total': [],
                'delay_sensitive': [],
                'reliability_sensitive': [],
                'throughput_sensitive': []
            },
            'packet_loss': {
                'total': [],
                'delay_sensitive': [],
                'reliability_sensitive': [],
                'throughput_sensitive': []
            },
            'throughput': {
                'total': [],
                'delay_sensitive': [],
                'reliability_sensitive': [],
                'throughput_sensitive': []
            },
            'qos_satisfaction': {
                'delay_sensitive': [],
                'reliability_sensitive': [],
                'throughput_sensitive': []
            }
        }
        
        # 获取QoS阈值
        self.qos_config = config['environment']['qos']
        self.delay_threshold = self.qos_config['delay_sensitive']['delay_threshold']
        self.loss_threshold = self.qos_config['reliability_sensitive']['loss_threshold']
        self.throughput_threshold = self.qos_config['throughput_sensitive']['throughput_threshold']
    
    def update(self, path_metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """
        更新性能指标
        
        Args:
            path_metrics: 各类型数据包的端到端性能指标
        """
        # 更新延迟指标
        for qos_type in path_metrics['delays']:
            if path_metrics['delays'][qos_type]:
                # 过滤掉无效的延迟值
                valid_delays = [d for d in path_metrics['delays'][qos_type] if d >= 0]
                if valid_delays:
                    self.metrics['delay'][qos_type].append(np.mean(valid_delays))
        
        # 更新丢包率指标
        for qos_type in path_metrics['packet_losses']:
            if path_metrics['packet_losses'][qos_type]:
                # 确保丢包率在[0,1]范围内
                valid_losses = [max(0.0, min(1.0, pl)) for pl in path_metrics['packet_losses'][qos_type]]
                if valid_losses:
                    self.metrics['packet_loss'][qos_type].append(np.mean(valid_losses))
        
        # 更新吞吐量指标
        for qos_type in path_metrics['throughputs']:
            if path_metrics['throughputs'][qos_type]:
                # 过滤掉无效的吞吐量值
                valid_throughputs = [t for t in path_metrics['throughputs'][qos_type] if t >= 0]
                if valid_throughputs:
                    self.metrics['throughput'][qos_type].append(np.mean(valid_throughputs))
        
        # 更新QoS满意度
        self._update_qos_satisfaction(path_metrics)
    
    def _update_qos_satisfaction(self, path_metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """
        更新QoS满意度
        
        Args:
            path_metrics: 各类型数据包的端到端性能指标
        """
        # 延迟敏感型流量的QoS满意度
        if path_metrics['delays']['delay_sensitive']:
            valid_delays = [d for d in path_metrics['delays']['delay_sensitive'] if d >= 0]
            if valid_delays:
                satisfaction = np.mean([
                    max(0.0, min(1.0, 1 - d / self.delay_threshold))
                    for d in valid_delays
                ])
                self.metrics['qos_satisfaction']['delay_sensitive'].append(satisfaction)
        
        # 可靠性敏感型流量的QoS满意度
        if path_metrics['packet_losses']['reliability_sensitive']:
            valid_losses = [max(0.0, min(1.0, pl)) for pl in path_metrics['packet_losses']['reliability_sensitive']]
            if valid_losses:
                satisfaction = np.mean([
                    max(0.0, min(1.0, 1 - pl / self.loss_threshold))
                    for pl in valid_losses
                ])
                self.metrics['qos_satisfaction']['reliability_sensitive'].append(satisfaction)
        
        # 吞吐量敏感型流量的QoS满意度
        if path_metrics['throughputs']['throughput_sensitive']:
            valid_throughputs = [t for t in path_metrics['throughputs']['throughput_sensitive'] if t >= 0]
            if valid_throughputs:
                satisfaction = np.mean([
                    max(0.0, min(1.0, t / self.throughput_threshold))
                    for t in valid_throughputs
                ])
                self.metrics['qos_satisfaction']['throughput_sensitive'].append(satisfaction)
    
    def get_average_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        获取平均性能指标
        
        Returns:
            Dict[str, Dict[str, float]]: 平均性能指标
        """
        averages = {}
        
        for metric_name, metric_data in self.metrics.items():
            averages[metric_name] = {}
            for qos_type, values in metric_data.items():
                if values:
                    averages[metric_name][qos_type] = float(np.mean(values))
                else:
                    averages[metric_name][qos_type] = 0.0
        
        return averages
    
    def get_latest_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        获取最新性能指标
        
        Returns:
            Dict[str, Dict[str, float]]: 最新性能指标
        """
        latest = {}
        
        for metric_name, metric_data in self.metrics.items():
            latest[metric_name] = {}
            for qos_type, values in metric_data.items():
                if values:
                    latest[metric_name][qos_type] = float(values[-1])
                else:
                    latest[metric_name][qos_type] = 0.0
        
        return latest
    
    def save_metrics(self, 
                    save_dir: str,
                    algorithm_name: Optional[str] = None) -> None:
        """
        保存性能指标
        
        Args:
            save_dir: 保存目录
            algorithm_name: 算法名称（可选）
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if algorithm_name:
            filename = f'metrics_{algorithm_name}_{timestamp}.json'
        else:
            filename = f'metrics_{timestamp}.json'
        
        # 保存指标
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'config': self.config
            }, f, indent=4)
    
    def load_metrics(self, filepath: str) -> None:
        """
        加载性能指标
        
        Args:
            filepath: 指标文件路径
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.metrics = data['metrics']
            self.config = data['config']
            
            # 更新QoS阈值
            self.qos_config = self.config['environment']['qos']
            self.delay_threshold = self.qos_config['delay_sensitive']['delay_threshold']
            self.loss_threshold = self.qos_config['reliability_sensitive']['loss_threshold']
            self.throughput_threshold = self.qos_config['throughput_sensitive']['throughput_threshold']
    
    def reset(self) -> None:
        """重置所有指标"""
        for metric_type in self.metrics:
            for qos_type in self.metrics[metric_type]:
                self.metrics[metric_type][qos_type].clear() 
 