import unittest
from unittest.mock import Mock, patch
from ..simulator import NetworkSimulator
from ..event import Event, EventType
import numpy as np

class TestNetworkSimulator(unittest.TestCase):
    """测试网络仿真器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'topology': {
                'orbital_planes': 1,  # 只使用一个轨道面
                'satellites_per_plane': 4,
                'altitude': 500,  # 降低轨道高度到500km
                'inclination': 0,  # 设置倾角为0，使卫星在赤道面上
                'link_bandwidth': 100,
                'link_delay': 0.01,
                'max_range': 5000,  # 最大通信距离5000km
                'link': {
                    'frequency': 26e9,  # Ka频段 26GHz
                    'transmit_power': 100.0,  # 发射功率100W
                    'noise_temperature': 290,  # 290K噪声温度
                    'bandwidth': 1e9,  # 1GHz带宽
                    'capacity': 1e9  # 直接设置链路容量为1Gbps
                }
            },
            'simulation': {
                'network': {
                    'total_satellites': 4,  # orbital_planes * satellites_per_plane
                    'buffer_size': 1000,
                    'orbital_planes': 1,
                    'satellites_per_plane': 4,
                    'max_range': 5000  # 最大通信距离(km)
                },
                'link': {
                    'buffer_size': 1000
                },
                'traffic': {
                    'packet_size': 1000,
                    'qos_classes': [
                        {
                            'name': 'delay_sensitive',
                            'delay_threshold': 50,  # 最大延迟阈值(ms)
                            'weight': 0.4
                        },
                        {
                            'name': 'reliability_sensitive',
                            'loss_threshold': 0.01,  # 最大丢包率阈值
                            'weight': 0.4
                        },
                        {
                            'name': 'throughput_sensitive',
                            'throughput_threshold': 100,  # 最小吞吐量阈值(Mbps)
                            'weight': 0.2
                        }
                    ]
                },
                'step_interval': 0.1
            }
        }
        self.simulator = NetworkSimulator(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(len(self.simulator.event_queue), 0)
        self.assertEqual(self.simulator.packet_counter, 0)
        self.assertIsNotNone(self.simulator.topology)
        
        # 检查指标初始化
        self.assertIn('delays', self.simulator.metrics)
        self.assertIn('losses', self.simulator.metrics)
        self.assertIn('queue_lengths', self.simulator.metrics)
        self.assertIn('link_states', self.simulator.metrics)
    
    def test_schedule_event(self):
        """测试事件调度"""
        event = Event(
            time=1.0,
            type=EventType.PACKET_ARRIVAL,
            data={'test': 'data'}
        )
        
        self.simulator.schedule_event(event)
        self.assertEqual(len(self.simulator.event_queue), 1)
        
        # 检查事件优先级排序
        event2 = Event(
            time=0.5,
            type=EventType.PACKET_ARRIVAL,
            data={'test': 'data2'}
        )
        self.simulator.schedule_event(event2)
        
        _, _, next_event = self.simulator.event_queue[0]
        self.assertEqual(next_event.time, 0.5)
    
    def test_run_simulation(self):
        """测试仿真运行"""
        # 创建测试事件
        event1 = Event(
            time=1.0,
            type=EventType.PACKET_ARRIVAL,
            data={
                'packet_id': 'test1',
                'src': 0,
                'dst': 1,
                'current_node': 0,
                'next_hop': 1,
                'size': 1000,
                'qos_class': 0,
                'creation_time': 0.0  # 添加创建时间
            }
        )
        event2 = Event(
            time=2.0,
            type=EventType.TOPOLOGY_UPDATE,
            data={
                'node_id': 0,
                'position': (0, 0, 0),
                'velocity': (1, 1, 1)
            }
        )
        
        self.simulator.schedule_event(event1)
        self.simulator.schedule_event(event2)
        
        # 运行仿真
        self.simulator.run(duration=3.0)
        
        # 检查仿真时间
        self.assertGreaterEqual(self.simulator.current_time, 2.0)
        
        # 检查事件队列是否为空
        self.assertEqual(len(self.simulator.event_queue), 0)
    
    def test_packet_handling(self):
        """测试数据包处理"""
        # 初始化拓扑
        self.simulator.topology.update(0.0)  # 更新拓扑以建立链路
        
        # 检查链路状态
        link_data = self.simulator.topology.get_link_data(0, 1)
        print(f"\nLink data between node 0 and 1: {link_data}")
        
        # 创建数据包到达事件
        event = Event(
            time=1.0,
            type=EventType.PACKET_ARRIVAL,
            data={
                'packet_id': 'test',
                'src': 0,
                'dst': 2,
                'current_node': 0,
                'next_hop': 1,  # 0和1是同一轨道面内的相邻卫星
                'size': 1000,
                'qos_class': 0,
                'creation_time': 0.5
            }
        )
        
        # 模拟路由算法
        self.simulator.routing_algorithm = Mock()
        self.simulator.routing_algorithm.get_next_hop.return_value = 1
        
        # 处理数据包
        self.simulator._handle_packet_arrival(event)
        
        # 检查是否生成了发送事件
        self.assertEqual(len(self.simulator.event_queue), 1)
        
        # 处理发送事件
        _, _, departure_event = self.simulator.event_queue[0]
        self.simulator._handle_packet_departure(departure_event)
        
        # 检查是否生成了到达事件
        self.assertEqual(len(self.simulator.event_queue), 2)
    
    def test_topology_update(self):
        """测试拓扑更新"""
        event = Event(
            time=1.0,
            type=EventType.TOPOLOGY_UPDATE,
            data={
                'node_id': 0,
                'position': (0, 0, 0),
                'velocity': (1, 1, 1)
            }
        )
        
        # 处理拓扑更新
        self.simulator._handle_topology_update(event)
        
        # 检查指标更新
        self.assertGreater(len(self.simulator.metrics['link_states']), 0)
        self.assertGreater(len(self.simulator.metrics['queue_lengths']), 0)
    
    def test_network_metrics(self):
        """测试网络性能指标"""
        # 初始化拓扑和网络状态
        self.simulator.topology.update(0.0)
        self.simulator._update_network_state()  # 确保网络状态被初始化
        
        # 创建一系列数据包事件来测试网络指标
        for i in range(5):
            # 创建数据包到达事件
            arrival_event = Event(
                time=float(i),
                type=EventType.PACKET_ARRIVAL,
                data={
                    'packet_id': f'test_packet_{i}',
                    'src': 0,
                    'dst': 2,
                    'current_node': 0,
                    'next_hop': 1,
                    'size': 1000,
                    'qos_class': 0,
                    'creation_time': float(i)
                }
            )
            
            # 模拟路由算法
            self.simulator.routing_algorithm = Mock()
            self.simulator.routing_algorithm.get_next_hop.return_value = 1
            
            # 处理数据包
            self.simulator._handle_packet_arrival(arrival_event)
            
            # 更新网络状态
            self.simulator._update_network_state()
        
        # 检查延迟指标
        self.assertGreater(len(self.simulator.metrics['delays']), 0)
        for delay in self.simulator.metrics['delays']:
            self.assertGreaterEqual(delay, 0)  # 延迟应该是非负的
        
        # 检查丢包指标
        initial_losses = len(self.simulator.metrics['losses'])
        
        # 创建一个会导致丢包的事件（通过设置无效链路）
        self.simulator.topology = Mock()
        self.simulator.topology.get_link_data.return_value = {
            'capacity': 0.0,  # 设置链路容量为0，使数据包被丢弃
            'delay': float('inf'),
            'distance': float('inf'),
            'quality': 0.0
        }
        
        drop_event = Event(
            time=6.0,
            type=EventType.PACKET_ARRIVAL,
            data={
                'packet_id': 'drop_test',
                'src': 0,
                'dst': 2,
                'current_node': 0,
                'next_hop': 1,
                'size': 1000,
                'qos_class': 0,
                'creation_time': 6.0
            }
        )
        
        # 处理可能导致丢包的数据包
        self.simulator._handle_packet_arrival(drop_event)
        
        # 验证丢包数增加
        self.assertGreater(len(self.simulator.metrics['losses']), initial_losses)
        
        # 检查队列长度指标
        self.assertGreater(len(self.simulator.metrics['queue_lengths']), 0)
        for queue_lengths in self.simulator.metrics['queue_lengths']:
            self.assertTrue(all(0 <= ql <= self.simulator.config['simulation']['link']['buffer_size'] 
                              for ql in queue_lengths))
        
        # 检查链路状态指标
        self.assertGreater(len(self.simulator.metrics['link_states']), 0)
        for link_states in self.simulator.metrics['link_states']:
            self.assertTrue(all(0 <= ls <= 1 for ls in link_states.flatten()))  # 链路状态应该在0-1之间
    
    def test_qos_metrics(self):
        """测试QoS相关指标"""
        # 初始化拓扑
        self.simulator.topology.update(0.0)
        
        # 创建不同QoS类型的数据包
        qos_classes = [0, 1, 2]  # 延迟敏感、可靠性敏感、吞吐量敏感
        
        for qos_class in qos_classes:
            # 设置当前数据包
            self.simulator.current_packet = {
                'source': 0,
                'destination': 2,
                'current_node': 0,
                'size': np.array([1000], dtype=np.float32),
                'qos_class': qos_class
            }
            
            event = Event(
                time=float(qos_class),
                type=EventType.PACKET_ARRIVAL,
                data={
                    'packet_id': f'qos_test_{qos_class}',
                    'src': 0,
                    'dst': 2,
                    'current_node': 0,
                    'next_hop': 1,
                    'size': 1000,
                    'qos_class': qos_class,
                    'creation_time': float(qos_class)
                }
            )
            
            # 模拟路由算法
            self.simulator.routing_algorithm = Mock()
            self.simulator.routing_algorithm.get_next_hop.return_value = 1
            
            # 处理数据包
            self.simulator._handle_packet_arrival(event)
            
            # 获取最新的网络状态
            info = self.simulator._get_info()
            
            # 检查QoS满意度指标
            self.assertIn('qos_satisfaction', info)
            qos_metrics = info['qos_satisfaction']
            
            # 验证QoS指标的合理性
            if qos_class == 0:  # 延迟敏感
                self.assertGreaterEqual(qos_metrics['delay_sensitive'], 0)
                self.assertLessEqual(qos_metrics['delay_sensitive'], 1)
            elif qos_class == 1:  # 可靠性敏感
                self.assertGreaterEqual(qos_metrics['reliability_sensitive'], 0)
                self.assertLessEqual(qos_metrics['reliability_sensitive'], 1)
            else:  # 吞吐量敏感
                self.assertGreaterEqual(qos_metrics['throughput_sensitive'], 0)
                self.assertLessEqual(qos_metrics['throughput_sensitive'], 1)

if __name__ == '__main__':
    unittest.main() 