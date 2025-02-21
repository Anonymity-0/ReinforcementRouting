import unittest
import numpy as np
import os
import json
from ..evaluation import Evaluator

class TestEvaluator(unittest.TestCase):
    """评估器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试配置
        cls.config = {
            'environment': {
                'qos': {
                    'delay_sensitive': {
                        'delay_threshold': 50
                    },
                    'reliability_sensitive': {
                        'loss_threshold': 0.001
                    },
                    'throughput_sensitive': {
                        'throughput_threshold': 100
                    }
                }
            }
        }
        
        # 创建评估器实例
        cls.evaluator = Evaluator(cls.config)
        
        # 创建测试数据
        cls.test_data = {
            'delays': {
                'total': [30, 40, 50],
                'delay_sensitive': [20, 30, 40],
                'reliability_sensitive': [35, 45, 55],
                'throughput_sensitive': [25, 35, 45]
            },
            'packet_losses': {
                'total': [0.001, 0.002, 0.003],
                'delay_sensitive': [0.0005, 0.0015, 0.0025],
                'reliability_sensitive': [0.0008, 0.0018, 0.0028],
                'throughput_sensitive': [0.0012, 0.0022, 0.0032]
            },
            'throughputs': {
                'total': [90, 100, 110],
                'delay_sensitive': [85, 95, 105],
                'reliability_sensitive': [95, 105, 115],
                'throughput_sensitive': [100, 110, 120]
            }
        }
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.evaluator.delay_threshold, 50)
        self.assertEqual(self.evaluator.loss_threshold, 0.001)
        self.assertEqual(self.evaluator.throughput_threshold, 100)
        
        for metric_type in ['delay', 'packet_loss', 'throughput']:
            self.assertIn(metric_type, self.evaluator.metrics)
            for qos_type in ['total', 'delay_sensitive', 
                           'reliability_sensitive', 'throughput_sensitive']:
                self.assertIn(qos_type, self.evaluator.metrics[metric_type])
    
    def test_update(self):
        """测试指标更新"""
        # 重置评估器
        self.evaluator.reset()
        
        # 更新指标
        self.evaluator.update(
            self.test_data['delays'],
            self.test_data['packet_losses'],
            self.test_data['throughputs']
        )
        
        # 检查延迟指标
        for qos_type in self.test_data['delays']:
            self.assertEqual(
                len(self.evaluator.metrics['delay'][qos_type]),
                1
            )
            self.assertAlmostEqual(
                self.evaluator.metrics['delay'][qos_type][0],
                np.mean(self.test_data['delays'][qos_type])
            )
        
        # 检查丢包率指标
        for qos_type in self.test_data['packet_losses']:
            self.assertEqual(
                len(self.evaluator.metrics['packet_loss'][qos_type]),
                1
            )
            self.assertAlmostEqual(
                self.evaluator.metrics['packet_loss'][qos_type][0],
                np.mean(self.test_data['packet_losses'][qos_type])
            )
        
        # 检查吞吐量指标
        for qos_type in self.test_data['throughputs']:
            self.assertEqual(
                len(self.evaluator.metrics['throughput'][qos_type]),
                1
            )
            self.assertAlmostEqual(
                self.evaluator.metrics['throughput'][qos_type][0],
                np.mean(self.test_data['throughputs'][qos_type])
            )
    
    def test_update_qos_satisfaction(self):
        """测试QoS满意度更新"""
        # 重置评估器
        self.evaluator.reset()
        
        # 更新指标
        self.evaluator.update(
            self.test_data['delays'],
            self.test_data['packet_losses'],
            self.test_data['throughputs']
        )
        
        # 检查延迟敏感型流量的QoS满意度
        delay_satisfaction = self.evaluator.metrics['qos_satisfaction']['delay_sensitive'][0]
        self.assertGreaterEqual(delay_satisfaction, 0)
        self.assertLessEqual(delay_satisfaction, 1)
        
        # 检查可靠性敏感型流量的QoS满意度
        reliability_satisfaction = self.evaluator.metrics['qos_satisfaction']['reliability_sensitive'][0]
        self.assertGreaterEqual(reliability_satisfaction, 0)
        self.assertLessEqual(reliability_satisfaction, 1)
        
        # 检查吞吐量敏感型流量的QoS满意度
        throughput_satisfaction = self.evaluator.metrics['qos_satisfaction']['throughput_sensitive'][0]
        self.assertGreaterEqual(throughput_satisfaction, 0)
        self.assertLessEqual(throughput_satisfaction, 1)
    
    def test_get_average_metrics(self):
        """测试获取平均指标"""
        # 重置评估器
        self.evaluator.reset()
        
        # 更新多次指标
        for _ in range(3):
            self.evaluator.update(
                self.test_data['delays'],
                self.test_data['packet_losses'],
                self.test_data['throughputs']
            )
        
        # 获取平均指标
        averages = self.evaluator.get_average_metrics()
        
        # 检查返回值结构
        self.assertIn('delay', averages)
        self.assertIn('packet_loss', averages)
        self.assertIn('throughput', averages)
        self.assertIn('qos_satisfaction', averages)
        
        # 检查数值
        for metric_type in averages:
            for qos_type in averages[metric_type]:
                self.assertIsInstance(averages[metric_type][qos_type], float)
    
    def test_get_latest_metrics(self):
        """测试获取最新指标"""
        # 重置评估器
        self.evaluator.reset()
        
        # 更新多次指标
        for _ in range(3):
            self.evaluator.update(
                self.test_data['delays'],
                self.test_data['packet_losses'],
                self.test_data['throughputs']
            )
        
        # 获取最新指标
        latest = self.evaluator.get_latest_metrics()
        
        # 检查返回值结构
        self.assertIn('delay', latest)
        self.assertIn('packet_loss', latest)
        self.assertIn('throughput', latest)
        self.assertIn('qos_satisfaction', latest)
        
        # 检查数值
        for metric_type in latest:
            for qos_type in latest[metric_type]:
                self.assertIsInstance(latest[metric_type][qos_type], float)
    
    def test_save_load_metrics(self):
        """测试指标保存和加载"""
        # 重置评估器
        self.evaluator.reset()
        
        # 更新指标
        self.evaluator.update(
            self.test_data['delays'],
            self.test_data['packet_losses'],
            self.test_data['throughputs']
        )
        
        # 保存指标
        save_dir = 'test_metrics'
        os.makedirs(save_dir, exist_ok=True)
        self.evaluator.save_metrics(save_dir, 'test')
        
        # 查找保存的文件
        files = os.listdir(save_dir)
        metric_file = None
        for f in files:
            if f.startswith('metrics_test_') and f.endswith('.json'):
                metric_file = f
                break
        
        self.assertIsNotNone(metric_file)
        
        # 创建新的评估器
        new_evaluator = Evaluator(self.config)
        
        # 加载指标
        new_evaluator.load_metrics(os.path.join(save_dir, metric_file))
        
        # 比较指标
        for metric_type in self.evaluator.metrics:
            for qos_type in self.evaluator.metrics[metric_type]:
                self.assertEqual(
                    len(self.evaluator.metrics[metric_type][qos_type]),
                    len(new_evaluator.metrics[metric_type][qos_type])
                )
                if len(self.evaluator.metrics[metric_type][qos_type]) > 0:
                    self.assertAlmostEqual(
                        self.evaluator.metrics[metric_type][qos_type][0],
                        new_evaluator.metrics[metric_type][qos_type][0]
                    )
        
        # 清理测试文件
        os.remove(os.path.join(save_dir, metric_file))
        os.rmdir(save_dir)
    
    def test_reset(self):
        """测试重置"""
        # 更新指标
        self.evaluator.update(
            self.test_data['delays'],
            self.test_data['packet_losses'],
            self.test_data['throughputs']
        )
        
        # 重置
        self.evaluator.reset()
        
        # 检查所有指标列表是否为空
        for metric_type in self.evaluator.metrics:
            for qos_type in self.evaluator.metrics[metric_type]:
                self.assertEqual(len(self.evaluator.metrics[metric_type][qos_type]), 0)

if __name__ == '__main__':
    unittest.main() 
 