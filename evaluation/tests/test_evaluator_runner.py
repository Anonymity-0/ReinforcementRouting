import unittest
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch
import yaml

from ..evaluator_runner import EvaluatorRunner
from ..evaluation import Evaluator
from visualization.visualizer import Visualizer
from environment.network_environment import NetworkEnvironment

class TestEvaluatorRunner(unittest.TestCase):
    """评估运行器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试配置
        cls.config = {
            'environment': {
                'satellite': {
                    'total_satellites': 66,
                    'orbital_planes': 6,
                    'satellites_per_plane': 11,
                    'altitude': 550,
                    'inclination': 53.0,
                    'phase_shift': 0.0
                },
                'link': {
                    'bandwidth': 1e9,
                    'transmit_power': 10.0,
                    'noise_temperature': 290.0,
                    'buffer_size': 250,
                    'update_interval': 1.0
                },
                'traffic': {
                    'packet_size': 1024,
                    'poisson_lambda': 100
                },
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
            },
            'evaluation': {
                'eval_episodes': 2
            },
            'visualization': {
                'plot_topology': True,
                'plot_metrics': True,
                'plot_qos': True,
                'interactive': False,
                'save_plots': True,
                'update_interval': 1.0
            }
        }
        
        # 保存测试配置到临时文件
        cls.config_path = 'test_config.yaml'
        with open(cls.config_path, 'w') as f:
            yaml.dump(cls.config, f)
        
        # 创建测试目录
        cls.test_dir = 'test_results'
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # 创建模拟的logger
        cls.mock_logger = MagicMock()
        
        # 创建环境和评估器实例
        cls.env = NetworkEnvironment(cls.config_path)
        cls.evaluator = Evaluator(cls.config)
        cls.visualizer = Visualizer(cls.config)
        
        # 创建模拟的算法
        cls.mock_algorithm = MagicMock()
        cls.mock_algorithm.act.return_value = 1
        
        # 创建评估运行器实例
        cls.runner = EvaluatorRunner(
            cls.config,
            cls.env,
            cls.mock_algorithm,
            cls.evaluator,
            cls.visualizer,
            cls.mock_logger,
            cls.test_dir
        )
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试文件和目录
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.runner.config, self.config)
        self.assertEqual(self.runner.env, self.env)
        self.assertEqual(self.runner.algorithm, self.mock_algorithm)
        self.assertEqual(self.runner.evaluator, self.evaluator)
        self.assertEqual(self.runner.visualizer, self.visualizer)
        self.assertEqual(self.runner.logger, self.mock_logger)
        self.assertEqual(self.runner.save_dir, self.test_dir)
    
    def test_collect_metrics(self):
        """测试指标收集"""
        # 重置环境
        state = self.env.reset()[0]
        
        # 收集指标
        metrics = self.runner.collect_metrics(0, 'delay_sensitive')
        
        # 检查返回值结构
        self.assertIsInstance(metrics, dict)
        for neighbor, values in metrics.items():
            self.assertIsInstance(neighbor, int)
            self.assertIn('delay', values)
            self.assertIn('packet_loss', values)
            self.assertIn('throughput', values)
            
            self.assertIsInstance(values['delay'], float)
            self.assertIsInstance(values['packet_loss'], float)
            self.assertIsInstance(values['throughput'], float)
    
    def test_run_evaluation(self):
        """测试评估运行"""
        # 运行评估
        results = self.runner.run_evaluation()
        
        # 检查返回值结构
        self.assertIn('mean_reward', results)
        self.assertIn('std_reward', results)
        self.assertIn('mean_success_rate', results)
        self.assertIn('mean_delay', results)
        self.assertIn('mean_packet_loss', results)
        self.assertIn('mean_throughput', results)
        self.assertIn('mean_qos_satisfaction', results)
        
        # 检查结果文件是否生成
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'evaluation_results.json')))
        
        # 检查日志记录
        self.mock_logger.info.assert_called()
    
    def test_save_results(self):
        """测试结果保存"""
        # 创建测试数据
        results = {
            'episode_rewards': [1.0, 2.0],
            'success_rate': [0.5, 0.6],
            'average_delay': [30.0, 35.0],
            'average_packet_loss': [0.001, 0.002],
            'average_throughput': [100.0, 110.0],
            'qos_satisfaction': [0.8, 0.9]
        }
        
        final_results = {
            'mean_reward': 1.5,
            'std_reward': 0.5,
            'mean_success_rate': 0.55,
            'mean_delay': 32.5,
            'mean_packet_loss': 0.0015,
            'mean_throughput': 105.0,
            'mean_qos_satisfaction': 0.85
        }
        
        # 保存结果
        self.runner.save_results(results, final_results)
        
        # 检查文件是否生成
        results_file = os.path.join(self.test_dir, 'evaluation_results.json')
        self.assertTrue(os.path.exists(results_file))
        
        # 检查文件内容
        with open(results_file, 'r') as f:
            saved_data = json.load(f)
            self.assertIn('episode_results', saved_data)
            self.assertIn('final_results', saved_data)
            self.assertEqual(saved_data['final_results'], final_results)
    
    @patch('visualization.visualizer.Visualizer.plot_performance_metrics')
    @patch('visualization.visualizer.Visualizer.plot_qos_comparison')
    def test_plot_results(self, mock_plot_qos, mock_plot_metrics):
        """测试结果绘制"""
        # 绘制结果
        self.runner.plot_results()
        
        # 检查是否调用了绘图函数
        self.assertEqual(mock_plot_metrics.call_count, 3)  # delay, packet_loss, throughput
        mock_plot_qos.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
 