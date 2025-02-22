import unittest
import numpy as np
import torch
import os
import shutil
from ..mappo.mappo_routing import MAPPORouting
from ..mappo.networks import MAPPOActor, MAPPOCritic

class TestMAPPORouting(unittest.TestCase):
    """MAPPO路由算法测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试配置
        cls.config = {
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
                    'use_reward_normalization': True,
                    'use_advantage_normalization': True,
                    'use_huber_loss': True,
                    'use_value_clip': True,
                    'value_clip_param': 0.2
                }
            }
        }
        
        # 创建算法实例
        cls.algorithm = MAPPORouting(cls.config)
        
        # 创建测试状态
        cls.state = {
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
        self.assertEqual(self.algorithm.num_agents, 66)
        self.assertEqual(len(self.algorithm.actors), 66)
        self.assertIsInstance(self.algorithm.critic, MAPPOCritic)
        self.assertEqual(len(self.algorithm.actor_optimizers), 66)
        self.assertIsInstance(self.algorithm.critic_optimizer, torch.optim.Optimizer)
    
    def test_reset_buffers(self):
        """测试缓冲区重置"""
        self.algorithm.reset_buffers()
        
        for i in range(self.algorithm.num_agents):
            self.assertEqual(len(self.algorithm.states[i]), 0)
            self.assertEqual(len(self.algorithm.actions[i]), 0)
            self.assertEqual(len(self.algorithm.rewards[i]), 0)
            self.assertEqual(len(self.algorithm.values[i]), 0)
            self.assertEqual(len(self.algorithm.log_probs[i]), 0)
            self.assertEqual(len(self.algorithm.masks[i]), 0)
    
    def test_preprocess_state(self):
        """测试状态预处理"""
        state_tensor = self.algorithm._preprocess_state(self.state)
        
        self.assertIsInstance(state_tensor, torch.Tensor)
        self.assertEqual(state_tensor.shape[0], self.algorithm.state_dim)
        self.assertEqual(state_tensor.device, self.algorithm.device)
    
    def test_get_global_state(self):
        """测试全局状态获取"""
        global_state = self.algorithm._get_global_state(self.state)
        
        self.assertIsInstance(global_state, torch.Tensor)
        expected_dim = self.algorithm.state_dim * self.algorithm.num_agents
        self.assertEqual(global_state.shape[0], expected_dim)
        self.assertEqual(global_state.device, self.algorithm.device)
    
    def test_get_next_hop(self):
        """测试下一跳选择"""
        # 测试确定性策略
        action = self.algorithm.get_next_hop(0, 65, self.state, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 66)
        
        # 测试随机策略
        action = self.algorithm.get_next_hop(0, 65, self.state, deterministic=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 66)
    
    def test_act(self):
        """测试动作选择"""
        # 测试训练模式
        self.algorithm.training = True
        action, info = self.algorithm.act(self.state, deterministic=False)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(info, dict)
        self.assertIn('value', info)
        self.assertIn('action_probs', info)
        
        # 测试评估模式
        self.algorithm.training = False
        action, info = self.algorithm.act(self.state, deterministic=True)
        
        self.assertIsInstance(action, int)
        self.assertIsNone(info)
    
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
        
        # 收集一些经验
        for _ in range(batch_size):
            action, _ = self.algorithm.act(self.state, deterministic=False)
            reward = np.random.rand()
            for agent_id in range(self.algorithm.num_agents):
                self.algorithm.rewards[agent_id].append(reward)
        
        # 更新算法
        info = self.algorithm.update(states, actions, rewards, next_states, dones)
        
        # 检查更新信息
        self.assertIn('actor_loss', info)
        self.assertIn('critic_loss', info)
        self.assertIn('entropy', info)
        self.assertIsInstance(info['actor_loss'], float)
        self.assertIsInstance(info['critic_loss'], float)
        self.assertIsInstance(info['entropy'], float)
        
        # 检查缓冲区是否已清空
        for agent_id in range(self.algorithm.num_agents):
            self.assertEqual(len(self.algorithm.states[agent_id]), 0)
            self.assertEqual(len(self.algorithm.actions[agent_id]), 0)
            self.assertEqual(len(self.algorithm.rewards[agent_id]), 0)
            self.assertEqual(len(self.algorithm.values[agent_id]), 0)
            self.assertEqual(len(self.algorithm.log_probs[agent_id]), 0)
            self.assertEqual(len(self.algorithm.masks[agent_id]), 0)
    
    def test_compute_advantages(self):
        """测试优势函数计算"""
        rewards = [1.0, 0.5, 0.8]
        values = [0.9, 0.4, 0.7]
        masks = [1.0, 1.0, 0.0]
        
        advantages = self.algorithm._compute_advantages(rewards, values, masks)
        
        self.assertIsInstance(advantages, torch.Tensor)
        self.assertEqual(len(advantages), len(rewards))
        self.assertEqual(advantages.device, self.algorithm.device)
    
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
            new_algorithm = MAPPORouting(self.config)
            
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

if __name__ == '__main__':
    unittest.main() 