import unittest
import numpy as np
import os
import shutil
import yaml
import torch

from algorithms.ppo.ppo_algorithm import PPOAlgorithm
from algorithms.ppo.ppo_routing import PPORouting

class TestPPOAlgorithm(unittest.TestCase):
    """测试PPO算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.config = {
            'algorithms': {
                'state_dim': 77,  # 1 + 1 + 1 + 66 + 66
                'action_dim': 60,
                'hidden_dim': 256,
                'ppo': {
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'ppo_epoch': 10,
                    'batch_size': 32,
                    'value_loss_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5
                }
            },
            'environment': {
                'satellite': {
                    'total_satellites': 66,
                    'orbit_height': 1500,
                    'min_elevation': 25
                }
            }
        }
        
        # 创建算法实例
        self.algorithm = PPOAlgorithm(self.config)
        
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
        self.assertEqual(self.algorithm.state_dim, 77)
        self.assertEqual(self.algorithm.action_dim, 60)
        self.assertEqual(self.algorithm.hidden_dim, 256)
        self.assertIsInstance(self.algorithm.actor, torch.nn.Module)
        self.assertIsInstance(self.algorithm.critic, torch.nn.Module)
    
    def test_preprocess_state(self):
        """测试状态预处理"""
        state_tensor = self.algorithm._preprocess_state(self.state)
        
        self.assertIsInstance(state_tensor, torch.Tensor)
        self.assertEqual(state_tensor.shape, (1, 77))
        self.assertEqual(state_tensor.device, self.algorithm.device)
    
    def test_compute_advantages(self):
        """测试优势函数计算"""
        rewards = [1.0, 0.5, 0.8]
        values = [0.9, 0.4, 0.7]
        masks = [1.0, 1.0, 0.0]
        
        advantages = self.algorithm._compute_advantages(rewards, values, masks)
        
        self.assertIsInstance(advantages, torch.Tensor)
        self.assertEqual(len(advantages), len(rewards))
        self.assertEqual(advantages.device, self.algorithm.device)
    
    def test_act(self):
        """测试动作选择"""
        # 测试随机策略
        action, info = self.algorithm.act(self.state, deterministic=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.algorithm.action_dim)
        self.assertIsInstance(info, dict)
        self.assertIn('value', info)
        self.assertIn('action_probs', info)
        
        # 测试确定性策略
        action, info = self.algorithm.act(self.state, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.algorithm.action_dim)
    
    def test_update(self):
        """测试算法更新"""
        # 收集一些经验
        batch_size = 4
        for _ in range(batch_size):
            action, _ = self.algorithm.act(self.state, deterministic=False)
            reward = np.random.rand()
            self.algorithm.rewards.append(reward)
        
        # 创建批次数据
        states = {k: np.stack([v for _ in range(batch_size)]) 
                 for k, v in self.state.items()}
        actions = np.random.randint(0, 66, size=batch_size)
        rewards = np.array([self.algorithm.rewards[i] for i in range(batch_size)])
        next_states = {k: np.stack([v for _ in range(batch_size)]) 
                      for k, v in self.state.items()}
        dones = np.zeros(batch_size)
        
        # 更新算法
        info = self.algorithm.update(states, actions, rewards, next_states, dones)
        
        # 检查训练信息
        self.assertIn('actor_loss', info)
        self.assertIn('critic_loss', info)
        self.assertIn('entropy', info)
        self.assertIsInstance(info['actor_loss'], float)
        self.assertIsInstance(info['critic_loss'], float)
        self.assertIsInstance(info['entropy'], float)
        
        # 检查经验缓冲区是否已清空
        self.assertEqual(len(self.algorithm.states), 0)
        self.assertEqual(len(self.algorithm.actions), 0)
        self.assertEqual(len(self.algorithm.rewards), 0)
        self.assertEqual(len(self.algorithm.values), 0)
        self.assertEqual(len(self.algorithm.log_probs), 0)
        self.assertEqual(len(self.algorithm.masks), 0)
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 创建临时目录
        test_dir = 'test_models'
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 保存模型
            self.algorithm.save(test_dir)
            
            # 检查文件是否存在
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'actor.pth')))
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'critic.pth')))
            
            # 创建新的算法实例
            new_algorithm = PPOAlgorithm(self.config)
            
            # 加载模型
            new_algorithm.load(test_dir)
            
            # 比较模型参数
            for p1, p2 in zip(self.algorithm.actor.parameters(),
                            new_algorithm.actor.parameters()):
                self.assertTrue(torch.equal(p1, p2))
            
            for p1, p2 in zip(self.algorithm.critic.parameters(),
                            new_algorithm.critic.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        
        finally:
            # 清理临时目录
            shutil.rmtree(test_dir)

class TestPPORouting(unittest.TestCase):
    """测试PPO路由算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.config = {
            'algorithms': {
                'state_dim': 77,  # 1 + 1 + 1 + 66 + 66
                'action_dim': 60,
                'hidden_dim': 256,
                'ppo': {
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'ppo_epoch': 10,
                    'batch_size': 32,
                    'value_loss_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5
                }
            },
            'environment': {
                'satellite': {
                    'total_satellites': 66,
                    'orbit_height': 1500,
                    'min_elevation': 25
                }
            }
        }
        
        # 创建路由算法实例
        self.routing = PPORouting(self.config)
        
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
        self.assertEqual(self.routing.name, "ppo")
        self.assertIsInstance(self.routing.algorithm, PPOAlgorithm)
        self.assertTrue(self.routing.training)
    
    def test_get_next_hop(self):
        """测试下一跳选择"""
        # 测试确定性策略
        action = self.routing.get_next_hop(0, 65, self.state, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 66)
        
        # 测试随机策略
        action = self.routing.get_next_hop(0, 65, self.state, deterministic=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 66)
    
    def test_act(self):
        """测试动作选择"""
        # 测试训练模式
        self.routing.training = True
        action, info = self.routing.act(self.state, deterministic=False)
        self.assertIsInstance(action, int)
        self.assertIsInstance(info, dict)
        self.assertIn('value', info)
        self.assertIn('action_probs', info)
        
        # 测试评估模式
        self.routing.training = False
        action, info = self.routing.act(self.state, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertIsInstance(info, dict)
    
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
        
        # 测试训练模式
        self.routing.training = True
        info = self.routing.update(states, actions, rewards, next_states, dones)
        self.assertIsInstance(info, dict)
        self.assertIn('actor_loss', info)
        self.assertIn('critic_loss', info)
        self.assertIn('entropy', info)
        
        # 测试评估模式
        self.routing.training = False
        info = self.routing.update(states, actions, rewards, next_states, dones)
        self.assertEqual(info, {})
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 创建临时目录
        test_dir = 'test_models'
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 保存模型
            self.routing.save(test_dir)
            
            # 检查文件是否存在
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'actor.pth')))
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'critic.pth')))
            
            # 创建新的路由算法实例
            new_routing = PPORouting(self.config)
            
            # 加载模型
            new_routing.load(test_dir)
            
            # 比较模型参数
            for p1, p2 in zip(self.routing.algorithm.actor.parameters(),
                            new_routing.algorithm.actor.parameters()):
                self.assertTrue(torch.equal(p1, p2))
            
            for p1, p2 in zip(self.routing.algorithm.critic.parameters(),
                            new_routing.algorithm.critic.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        
        finally:
            # 清理临时目录
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main() 
 