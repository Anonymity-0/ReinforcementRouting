import unittest
import numpy as np
import os
import shutil

from algorithms.dijkstra.dijkstra_algorithm import DijkstraAlgorithm
from algorithms.dijkstra.dijkstra_routing import DijkstraRouting

class TestDijkstraAlgorithm(unittest.TestCase):
    """测试Dijkstra算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.config = {
            'algorithms': {
                'state_dim': 73,
                'action_dim': 66,
                'hidden_dim': 256
            }
        }
        
        # 创建算法实例
        self.algorithm = DijkstraAlgorithm(self.config)
        
        # 创建测试状态
        self.state = {
            'topology': {
                'positions': np.random.randn(66, 3).astype(np.float32),
                'velocities': np.random.randn(66, 3).astype(np.float32)
            },
            'network': {
                'queue_lengths': np.random.rand(66).astype(np.float32),
                'link_states': np.random.rand(66, 66).astype(np.float32)
            },
            'packet': {
                'source': 0,
                'destination': 65,
                'current_node': 0,
                'size': np.array([1500]).astype(np.float32),
                'qos_class': 0
            }
        }
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.algorithm.num_nodes, 66)
    
    def test_get_next_hop(self):
        """测试下一跳选择"""
        # 测试有效路径
        next_hop = self.algorithm.get_next_hop(0, 65, self.state)
        self.assertIsInstance(next_hop, int)
        self.assertGreaterEqual(next_hop, 0)
        self.assertLess(next_hop, 66)
        
        # 测试相同源目的节点
        next_hop = self.algorithm.get_next_hop(0, 0, self.state)
        self.assertEqual(next_hop, 0)
        
        # 测试无效路径
        # 创建一个断开的网络
        disconnected_state = self.state.copy()
        disconnected_state['network'] = {
            'queue_lengths': np.zeros(66).astype(np.float32),
            'link_states': np.zeros((66, 66)).astype(np.float32)
        }
        next_hop = self.algorithm.get_next_hop(0, 65, disconnected_state)
        self.assertEqual(next_hop, -1)

class TestDijkstraRouting(unittest.TestCase):
    """测试Dijkstra路由算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.config = {
            'algorithms': {
                'state_dim': 73,
                'action_dim': 66,
                'hidden_dim': 256
            }
        }
        
        # 创建路由算法实例
        self.routing = DijkstraRouting(self.config)
        
        # 创建测试状态
        self.state = {
            'topology': {
                'positions': np.random.randn(66, 3).astype(np.float32),
                'velocities': np.random.randn(66, 3).astype(np.float32)
            },
            'network': {
                'queue_lengths': np.random.rand(66).astype(np.float32),
                'link_states': np.random.rand(66, 66).astype(np.float32)
            },
            'packet': {
                'source': 0,
                'destination': 65,
                'current_node': 0,
                'size': np.array([1500]).astype(np.float32),
                'qos_class': 0
            }
        }
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.routing.name, "dijkstra")
        self.assertIsInstance(self.routing.algorithm, DijkstraAlgorithm)
        self.assertFalse(self.routing.training)
    
    def test_get_next_hop(self):
        """测试下一跳选择"""
        # 测试有效路径
        next_hop = self.routing.get_next_hop(0, 65, self.state)
        self.assertIsInstance(next_hop, int)
        self.assertGreaterEqual(next_hop, 0)
        self.assertLess(next_hop, 66)
        
        # 测试相同源目的节点
        next_hop = self.routing.get_next_hop(0, 0, self.state)
        self.assertEqual(next_hop, 0)
        
        # 测试无效路径
        disconnected_state = self.state.copy()
        disconnected_state['network'] = {
            'queue_lengths': np.zeros(66).astype(np.float32),
            'link_states': np.zeros((66, 66)).astype(np.float32)
        }
        next_hop = self.routing.get_next_hop(0, 65, disconnected_state)
        self.assertEqual(next_hop, -1)
    
    def test_act(self):
        """测试动作选择"""
        # 测试动作选择
        action, info = self.routing.act(self.state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 66)
        self.assertEqual(info, {})
    
    def test_update(self):
        """测试算法更新"""
        # 创建批次数据
        batch_size = 4
        states = {k: np.stack([v for _ in range(batch_size)]) 
                 for k, v in self.state.items()}
        actions = np.random.randint(0, 66, size=batch_size)
        rewards = np.random.rand(batch_size)
        next_states = {k: np.stack([v for _ in range(batch_size)]) 
                      for k, v in self.state.items()}
        dones = np.zeros(batch_size)
        
        # Dijkstra算法不需要更新
        info = self.routing.update(states, actions, rewards, next_states, dones)
        self.assertEqual(info, {})
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 创建临时目录
        test_dir = 'test_models'
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 保存模型 (Dijkstra算法不需要保存模型)
            self.routing.save(test_dir)
            
            # 创建新的路由算法实例
            new_routing = DijkstraRouting(self.config)
            
            # 加载模型 (Dijkstra算法不需要加载模型)
            new_routing.load(test_dir)
            
            # 比较算法实例
            self.assertEqual(self.routing.name, new_routing.name)
            self.assertEqual(self.routing.training, new_routing.training)
        
        finally:
            # 清理临时目录
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main() 
 