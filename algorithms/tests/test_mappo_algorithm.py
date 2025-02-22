import unittest
import numpy as np
import torch
import os
import shutil
from algorithms.mappo.mappo_algorithm import MappoAlgorithm
from algorithms.mappo.mappo_routing import MAPPORouting

class TestMAPPOAlgorithm(unittest.TestCase):
    """测试MAPPO算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.config = {
            'algorithms': {
                'state_dim': 84,  # 3(pos) + 3(vel) + 1(queue) + 66(links) + 1(packet_size) + 1(current_node) + 1(destination) + 3(dest_pos) + 3(dest_vel) + 1(dest_queue)
                'action_dim': 66,
                'hidden_dim': 256,
                'mappo': {
                    'num_agents': 66,
                    'learning_rate': 3.0e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'ppo_epoch': 10,
                    'batch_size': 64,
                    'value_loss_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5,
                    'buffer_size': 2048,
                    'use_centralized_critic': True,
                    'use_shared_policy': True,
                    'use_global_state': True,
                    'cooperation_weight': 0.5,
                    'load_balance_weight': 0.3
                }
            }
        }
        
        # 创建算法实例
        self.algorithm = MappoAlgorithm(self.config)
        
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
        self.assertEqual(self.algorithm.state_dim, 84)
        self.assertEqual(self.algorithm.action_dim, 66)
        self.assertEqual(self.algorithm.hidden_dim, 256)
        self.assertIsInstance(self.algorithm.actors[0], torch.nn.Module)
        self.assertIsInstance(self.algorithm.critic, torch.nn.Module)
        self.assertEqual(self.algorithm.cooperation_weight, 0.5)
        self.assertEqual(self.algorithm.load_balance_weight, 0.3)
    
    def test_preprocess_state(self):
        """测试状态预处理"""
        state_tensor = self.algorithm._preprocess_state(self.state)
        
        self.assertIsInstance(state_tensor, torch.Tensor)
        self.assertEqual(state_tensor.shape, (1, 84))
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
        action, info = self.algorithm.act(self.state, 0, deterministic=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.algorithm.action_dim)
        self.assertIsInstance(info, dict)
        self.assertIn('value', info)
        self.assertIn('log_prob', info)
        
        # 测试确定性策略
        action, info = self.algorithm.act(self.state, 0, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.algorithm.action_dim)
    
    def test_update(self):
        """测试算法更新"""
        # 收集一些经验
        batch_size = 4
        for _ in range(batch_size):
            action, _ = self.algorithm.act(self.state, 0, deterministic=False)
            reward = np.random.rand()
            self.algorithm.rewards[0].append(reward)
        
        # 创建批次数据
        states = {k: np.stack([v for _ in range(batch_size)]) 
                 for k, v in self.state.items()}
        actions = np.random.randint(0, 66, size=batch_size)
        rewards = np.array([self.algorithm.rewards[0][i] for i in range(batch_size)])
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
        self.assertEqual(len(self.algorithm.states[0]), 0)
        self.assertEqual(len(self.algorithm.actions[0]), 0)
        self.assertEqual(len(self.algorithm.rewards[0]), 0)
        self.assertEqual(len(self.algorithm.values[0]), 0)
        self.assertEqual(len(self.algorithm.log_probs[0]), 0)
        self.assertEqual(len(self.algorithm.masks[0]), 0)
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 创建临时目录
        test_dir = 'test_models'
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 保存模型
            self.algorithm.save(test_dir)
            
            # 检查文件是否存在
            for i in range(self.algorithm.num_agents):
                self.assertTrue(os.path.exists(os.path.join(test_dir, f'actor_{i}.pth')))
            self.assertTrue(os.path.exists(os.path.join(test_dir, 'critic.pth')))
            
            # 创建新的算法实例
            new_algorithm = MappoAlgorithm(self.config)
            
            # 加载模型
            new_algorithm.load(test_dir)
            
            # 比较模型参数
            for i in range(self.algorithm.num_agents):
                for p1, p2 in zip(self.algorithm.actors[i].parameters(),
                                new_algorithm.actors[i].parameters()):
                    self.assertTrue(torch.equal(p1, p2))
            
            for p1, p2 in zip(self.algorithm.critic.parameters(),
                            new_algorithm.critic.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        
        finally:
            # 清理临时目录
            shutil.rmtree(test_dir)
    
    def test_link_state_processing(self):
        """测试链路状态处理"""
        # 创建一个特定的链路状态
        self.state['network']['link_states'] = np.zeros((66, 66), dtype=np.float32)
        # 设置当前节点(0)的邻居连接
        self.state['network']['link_states'][0][1] = 1.0
        self.state['network']['link_states'][0][2] = 0.8
        self.state['network']['link_states'][0][3] = 0.6
        
        # 测试动作选择
        action, info = self.algorithm.act(self.state, 0, deterministic=True)
        
        # 验证选择的动作是否为最好的邻居节点
        self.assertIn(action, [1, 2, 3])
        
    def test_invalid_action_handling(self):
        """测试无效动作处理"""
        # 创建一个断开的网络状态
        self.state['network']['link_states'] = np.zeros((66, 66), dtype=np.float32)
        
        # 测试动作选择
        action, info = self.algorithm.act(self.state, 0, deterministic=True)
        
        # 验证在没有有效连接时的行为
        self.assertEqual(action, 0)  # 应该返回当前节点
        
    def test_state_preprocessing(self):
        """测试状态预处理"""
        state_tensor = self.algorithm._preprocess_state(self.state)
        
        # 验证预处理后的状态维度
        self.assertEqual(state_tensor.shape[1], 84)  # 新的状态维度
        
        # 验证状态张量的内容
        self.assertTrue(torch.is_tensor(state_tensor))
        self.assertEqual(state_tensor.dtype, torch.float32)

class TestMAPPORouting(unittest.TestCase):
    """测试MAPPO路由算法"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试配置
        self.config = {
            'algorithms': {
                'state_dim': 84,  # 3(pos) + 3(vel) + 1(queue) + 66(links) + 1(packet_size) + 1(current_node) + 1(destination) + 3(dest_pos) + 3(dest_vel) + 1(dest_queue)
                'action_dim': 66,
                'hidden_dim': 256,
                'mappo': {
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'ppo_epoch': 10,
                    'batch_size': 32,
                    'value_loss_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5,
                    'cooperation_weight': 0.5,
                    'load_balance_weight': 0.3,
                    'num_agents': 66,
                    'buffer_size': 1024,
                    'use_centralized_critic': False,
                    'use_shared_policy': True,
                    'use_global_state': False,
                    'use_reward_normalization': True,
                    'use_advantage_normalization': True,
                    'use_huber_loss': True,
                    'use_value_clip': True,
                    'value_clip_param': 0.2
                }
            }
        }
        
        # 创建路由算法实例
        self.routing = MAPPORouting(self.config)
        
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
        self.assertEqual(self.routing.name, "mappo")
        self.assertIsInstance(self.routing.algorithm, MappoAlgorithm)
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
        
        # 测试评估模式
        self.routing.training = False
        action, info = self.routing.act(self.state, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertIsNone(info)
    
    def test_update(self):
        """测试算法更新"""
        # 收集一些经验
        batch_size = 4
        for _ in range(batch_size):
            action, _ = self.routing.act(self.state, deterministic=False)
            reward = np.random.rand()
            self.routing.algorithm.rewards[0].append(reward)
        
        # 创建批次数据
        states = {k: np.stack([v for _ in range(batch_size)]) 
                 for k, v in self.state.items()}
        actions = np.random.randint(0, 66, size=batch_size)
        rewards = np.array([self.routing.algorithm.rewards[0][i] for i in range(batch_size)])
        next_states = {k: np.stack([v for _ in range(batch_size)]) 
                      for k, v in self.state.items()}
        dones = np.zeros(batch_size)
        
        # 更新算法
        info = self.routing.update(states, actions, rewards, next_states, dones)
        
        # 检查训练信息
        self.assertIn('actor_loss', info)
        self.assertIn('critic_loss', info)
        self.assertIn('entropy', info)
        self.assertIsInstance(info['actor_loss'], float)
        self.assertIsInstance(info['critic_loss'], float)
        self.assertIsInstance(info['entropy'], float)
        
        # 检查经验缓冲区是否已清空
        self.assertEqual(len(self.routing.algorithm.states[0]), 0)
        self.assertEqual(len(self.routing.algorithm.actions[0]), 0)
        self.assertEqual(len(self.routing.algorithm.rewards[0]), 0)
        self.assertEqual(len(self.routing.algorithm.values[0]), 0)
        self.assertEqual(len(self.routing.algorithm.log_probs[0]), 0)
        self.assertEqual(len(self.routing.algorithm.masks[0]), 0)
    
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
            new_routing = MAPPORouting(self.config)
            
            # 加载模型
            new_routing.load(test_dir)
            
            # 比较模型参数
            for p1, p2 in zip(self.routing.algorithm.actors[0].parameters(),
                            new_routing.algorithm.actors[0].parameters()):
                self.assertTrue(torch.equal(p1, p2))
            
            for p1, p2 in zip(self.routing.algorithm.critic.parameters(),
                            new_routing.algorithm.critic.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        
        finally:
            # 清理临时目录
            shutil.rmtree(test_dir)