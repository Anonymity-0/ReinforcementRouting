import unittest
import numpy as np
from satellite_env import (
    SatelliteEnv, 
    ORBIT_HEIGHT_LEO, 
    NUM_ORBITS_LEO, 
    SATS_PER_ORBIT_LEO, 
    MAX_PATH_LENGTH,
    UPDATE_INTERVAL,
    DATA_GENERATION_RATE
)

class TestSatelliteEnv(unittest.TestCase):
    def setUp(self):
        self.env = SatelliteEnv()
        self.env.reset()

    def test_network_topology(self):
        """测试网络拓扑结构"""
        print("\n测试网络拓扑...")
        
        # 检查LEO节点数量
        expected_leo_count = NUM_ORBITS_LEO * SATS_PER_ORBIT_LEO
        actual_leo_count = len(self.env.leo_nodes)
        self.assertEqual(actual_leo_count, expected_leo_count, 
                        f"LEO节点数量不正确: 期望 {expected_leo_count}, 实际 {actual_leo_count}")
        print(f"✓ LEO节点数量正确: {actual_leo_count}")

        # 检查每个LEO的连接数
        for leo_name, leo_node in self.env.leo_nodes.items():
            connections = self.env.leo_neighbors[leo_name]
            self.assertGreaterEqual(len(connections), 2, 
                                  f"{leo_name} 连接数过少: {len(connections)}")
            self.assertLessEqual(len(connections), 4, 
                               f"{leo_name} 连接数过多: {len(connections)}")
        print("✓ 所有LEO节点的连接数在合理范围内(2-4)")

        # 验证轨道内连接
        for orbit in range(NUM_ORBITS_LEO):
            for pos in range(SATS_PER_ORBIT_LEO):
                current_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}'
                neighbors = self.env.leo_neighbors[current_leo]
                
                # 检查与前后卫星的连接
                if pos > 0:
                    prev_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos}'
                    self.assertIn(prev_leo, neighbors, 
                                f"{current_leo} 未与前一个卫星 {prev_leo} 连接")
                
                if pos < SATS_PER_ORBIT_LEO - 1:
                    next_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 2}'
                    self.assertIn(next_leo, neighbors, 
                                f"{current_leo} 未与后一个卫星 {next_leo} 连接")
        print("✓ 轨道内连接正确")

    def test_link_metrics(self):
        """测试链路性能指标"""
        print("\n测试链路性能指标...")
        
        # 选择两个相邻的LEO节点
        leo1 = 'leo1'
        leo2 = 'leo2'
        
        # 获取链路对象
        link = self.env.links_dict.get((leo1, leo2)) or self.env.links_dict.get((leo2, leo1))
        
        # 测试不同数据生成率下的性能
        test_rates = [5.0, 50.0, 500.0]
        
        for rate in test_rates:
            print(f"\n测试数据生成率: {rate:.2f} Gbps")
            
            # 重置链路状态
            link.packets = {
                'in_queue': set(),
                'processed': set(),
                'dropped': set(),
                'lost': set()
            }
            link.packet_timestamps = {}
            link.last_process_time = 0
            
            # 生成并发送数据包
            time_interval = 0.1  # 100ms
            self.env.current_data_rate = rate
            packets = self.env._generate_traffic_poisson(time_interval)
            print(f"生成的数据包数量: {packets}")
            
            # 添加数据包到队列
            accepted, dropped, _ = link.add_packets(packets, 0, self.env.simulation_time)
            print(f"接受的数据包: {len(accepted)}")
            print(f"丢弃的数据包: {len(dropped)}")
            
            # 给足够的处理时间
            self.env.simulation_time += UPDATE_INTERVAL
            processed = link.process_queue(self.env.simulation_time)
            print(f"处理的数据包: {len(processed)}")
            print(f"传输中丢失的数据包: {len(link.packets['lost'])}")
            
            # 计算总丢包率
            total_lost = len(dropped) + len(link.packets['lost'])
            total_loss_rate = (total_lost / packets * 100) if packets > 0 else 0
            print(f"总丢包率: {total_loss_rate:.2f}% ({total_lost}/{packets})")
            
            # 计算链路指标
            metrics = self.env._calculate_link_metrics(leo1, leo2)
            
            # 验证性能指标
            self.assertGreater(metrics['delay'], 0, "延迟应该大于0")
            self.assertGreater(metrics['bandwidth'], 0, "带宽应该大于0")
            self.assertGreaterEqual(metrics['loss'], 0, "丢包率应该大于等于0")
            self.assertLessEqual(metrics['loss'], 100, "丢包率不应超过100%")
            
            # 验证性能指标随数据生成率的变化
            if rate > 5.0:
                self.assertGreater(len(accepted), 0, "高数据率下应该有数据包被接受")
                self.assertGreater(len(processed), 0, "高数据率下应该有数据包被处理")
                self.assertGreater(metrics['delay'], link.base_delay, 
                                 "高数据率下延迟应该增加")
                # 验证总丢包率随数据生成率增加而增加
                self.assertGreater(total_loss_rate, 0, 
                                 "高数据率下应该有一定的丢包率")
            
            print(f"链路指标:")
            print(f"  - 延迟: {metrics['delay']:.2f} ms")
            print(f"  - 带宽: {metrics['bandwidth']:.2f} MHz")
            print(f"  - 丢包率: {metrics['loss']:.2f}%")

    def test_packet_transmission(self):
        """测试数据包传输"""
        print("\n测试数据包传输...")
        
        # 选择源和目标节点
        source = 'leo1'
        destination = 'leo2'
        
        # 测试数据包生成 - 减小生成的数据包数量
        packets = self.env._generate_traffic_poisson(time_interval=0.1)  # 减小时间间隔
        self.assertGreater(packets, 0, "应该生成数据包")
        print(f"✓ 生成了 {packets} 个数据包")

        # 获取链路
        link = self.env.links_dict.get((source, destination)) or \
               self.env.links_dict.get((destination, source))
        
        # 根据链路容量发送合理数量的数据包
        packets_to_send = min(packets, link.max_packets)  # 不超过链路最大容量
        accepted, dropped, _ = link.add_packets(packets_to_send, 0, self.env.simulation_time)
        
        # 验证接受率
        acceptance_rate = len(accepted) / packets_to_send * 100
        self.assertGreaterEqual(acceptance_rate, 50, "数据包接受率应该大于50%")
        print(f"✓ 成功接受 {len(accepted)} 个数据包 (接受率: {acceptance_rate:.2f}%)")

        # 给足够的处理时间
        process_time = self.env.simulation_time + (packets_to_send * link.base_delay)
        processed = link.process_queue(process_time)
        process_rate = len(processed) / len(accepted) * 100
        self.assertGreaterEqual(process_rate, 80, "数据包处理率应该大于80%")
        print(f"✓ 成功处理 {len(processed)} 个数据包 (处理率: {process_rate:.2f}%)")

        # 测试丢包计算
        metrics = self.env._calculate_link_metrics(source, destination)
        lost = link.calculate_packet_loss(processed, metrics['loss'])
        loss_rate = len(lost) / len(processed) * 100
        self.assertLessEqual(loss_rate, 20, "丢包率不应超过20%")
        print(f"✓ 丢失 {len(lost)} 个数据包 (丢包率: {loss_rate:.2f}%)")

    def test_cross_region_path_finding(self):
        """测试基于交叉区域的路径发现"""
        print("\n测试基于交叉区域的路径发现...")
        
        # 选择不同MEO区域的源和目标节点
        source_meo = 'meo1'
        dest_meo = 'meo2'
        
        # 找到属于这些MEO的LEO节点
        source = next(leo for leo, meo in self.env.leo_to_meo.items() if meo == source_meo)
        destination = next(leo for leo, meo in self.env.leo_to_meo.items() if meo == dest_meo)
        
        print(f"测试从 {source}({source_meo}) 到 {destination}({dest_meo}) 的路径")
        
        # 使用新的路径查找算法
        k = 3  # 查找前3条路径
        paths = self.env._find_k_shortest_paths_with_cross_region(source, destination, k, self.env.leo_graph)
        
        # 验证找到的路径
        self.assertIsNotNone(paths, "应该找到至少一条路径")
        self.assertGreater(len(paths), 0, "应该找到至少一条路径")
        self.assertLessEqual(len(paths), k * k, f"找到的路径数不应超过{k*k}条")
        
        # 验证每条路径
        for i, path in enumerate(paths):
            print(f"路径 {i+1}: {' -> '.join(path)}")
            
            # 验证路径的起点和终点
            self.assertEqual(path[0], source, "路径必须从源节点开始")
            self.assertEqual(path[-1], destination, "路径必须到达目标节点")
            
            # 验证路径的连续性
            for j in range(len(path)-1):
                current = path[j]
                next_node = path[j+1]
                self.assertIn(next_node, self.env.leo_neighbors[current], 
                             f"路径中的节点 {current} 和 {next_node} 之间没有连接")
            
            # 验证路径长度在合理范围内
            self.assertLess(len(path), MAX_PATH_LENGTH, 
                           f"路径 {i+1} 长度 ({len(path)}) 超过最大限制")
            
            # 验证路径经过交叉区域
            cross_region = False
            for j in range(len(path)-1):
                current_meo = self.env.leo_to_meo[path[j]]
                next_meo = self.env.leo_to_meo[path[j+1]]
                if current_meo != next_meo:
                    cross_region = True
                    break
            self.assertTrue(cross_region, "路径应该经过交叉区域")
            
            # 如果有多条路径，验证它们是不同的
            if i > 0:
                self.assertNotEqual(path, paths[i-1], "不同序号的路径应该是不同的")
        
        print(f"✓ 成功找到 {len(paths)} 条从 {source} 到 {destination} 的不同路径")

    def test_candidate_actions(self):
        """测试候选动作生成"""
        print("\n测试候选动作生成...")
        
        # 选择不同MEO区域的源和目标节点
        source_meo = 'meo1'
        dest_meo = 'meo2'
        source = next(leo for leo, meo in self.env.leo_to_meo.items() if meo == source_meo)
        destination = next(leo for leo, meo in self.env.leo_to_meo.items() if meo == dest_meo)
        
        # 获取可用动作
        available_actions = self.env.get_available_actions(source)
        
        # 获取候选动作
        candidate_actions = self.env.get_candidate_actions(source, destination, available_actions)
        
        # 验证候选动作
        self.assertIsNotNone(candidate_actions, "应该生成候选动作")
        self.assertGreater(len(candidate_actions), 0, "应该至少有一个候选动作")
        self.assertLessEqual(len(candidate_actions), len(available_actions), 
                            "候选动作数不应超过可用动作数")
        
        # 验证每个候选动作的有效性
        for action in candidate_actions:
            # 验证动作在可用动作范围内
            self.assertIn(action, available_actions, 
                         f"候选动作 {action} 不在可用动作列表中")
            
            # 获取对应的下一跳节点
            next_leo = list(self.env.leo_nodes.keys())[action]
            
            # 验证与当前节点的连接
            self.assertTrue(
                (source, next_leo) in self.env.links_dict or 
                (next_leo, source) in self.env.links_dict,
                f"候选动作 {action} 对应的节点 {next_leo} 与源节点 {source} 之间没有连接"
            )
        
        print(f"✓ 成功生成 {len(candidate_actions)} 个候选动作")


if __name__ == '__main__':
    unittest.main(verbosity=2) 