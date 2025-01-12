import numpy as np
import random
from collections import defaultdict, deque
from datetime import datetime
import math
import time


# 全局常量
ORBIT_HEIGHT_LEO = 1500
ORBIT_HEIGHT_MEO = 8000
NUM_ORBITS_LEO = 16
SATS_PER_ORBIT_LEO = 16
NUM_ORBITS_MEO = 2
SATS_PER_ORBIT_MEO = 8
INCLINATION = 55
EARTH_RADIUS = 6371
ALPHA = 0.1
GAMMA = 0.9
INITIAL_EPSILON = 0.9
MIN_EPSILON = 0.1
DECAY_RATE = 0.0001
MAX_PATH_LENGTH = 15
QUEUE_CAPACITY = 100  # MB
PACKET_SIZE = 25  # KB
DATA_GENERATION_RATE = 1.0  # Gbps
BANDWIDTH = 20  # MHz
SNR_MIN = 10  # dB
SNR_MAX = 30  # dB
UPDATE_INTERVAL = 100  # ms
TIME_STEP = 20  # ms
NETWORK_UPDATE_INTERVAL = 500  # ms

class Node:
    def __init__(self, name):
        self.name = name
        self.traffic = 0
        self.connected_satellites = set()

class Link:
    def __init__(self, node1, node2, delay, bandwidth, loss):
        self.node1 = node1
        self.node2 = node2
        self.base_delay = delay
        self.base_bandwidth = bandwidth
        self.base_loss = loss
        self.traffic = 0
        self.current_delay = delay
        self.current_bandwidth = bandwidth
        self.current_loss = loss
        self.weather_factor = 1.0
        self.last_process_time = 0
        self.last_update_time = 0
        self.max_packets = int((QUEUE_CAPACITY * 1024 * 1024) / (PACKET_SIZE * 1024))
        
        self.packets = {
            'in_queue': set(),
            'processed': set(),
            'dropped': set(),
            'lost': set()
        }
        self.packet_timestamps = {}

    def add_packets(self, num_packets, start_id, current_time):
        """添加数据包到队列"""
        
        accepted_packets = set()
        dropped_packets = set()
        
        for i in range(num_packets):
            packet_id = start_id + i
            if len(self.packets['in_queue']) < self.max_packets:
                self.packets['in_queue'].add(packet_id)
                self.packet_timestamps[packet_id] = current_time
                accepted_packets.add(packet_id)
            else:
                self.packets['dropped'].add(packet_id)
                dropped_packets.add(packet_id)
       
        self.traffic = len(self.packets['in_queue']) * PACKET_SIZE / 1024
        self.congestion_level = len(self.packets['in_queue']) / self.max_packets
        
        return accepted_packets, dropped_packets, start_id + num_packets

    def process_queue(self, current_time):
        """处理队列中的数据包"""
        if not self.packets['in_queue']:
            return set()
        
        time_delta = current_time - self.last_process_time
        if time_delta <= 0:
            return set()
       
        
        # 计算在给定时间内可以处理的数据包数量
        bits_per_packet = PACKET_SIZE * 8 * 1024  # 每个数据包的比特数
        available_bandwidth = self.current_bandwidth * 1e6  # 转换为 bps
        processable_bits = available_bandwidth * (time_delta / 1000.0)  # 可处理的总比特数
        packets_can_process = max(1, int(processable_bits / bits_per_packet))  # 确保至少处理1个包
        
        
        # 处理数据包
        processed_packets = set()
        packets_to_process = sorted(
            [(pid, self.packet_timestamps[pid]) for pid in self.packets['in_queue']],
            key=lambda x: x[1]
        )[:packets_can_process]
        
        # 增加基于队列长度的丢包概率
        queue_utilization = len(self.packets['in_queue']) / self.max_packets
        base_loss_rate = self.current_loss * (1 + queue_utilization)  # 根据队列利用率调整丢包率
        
        for packet_id, _ in packets_to_process:
            self.packets['in_queue'].remove(packet_id)
            # 根据丢包率决定是否丢弃数据包
            if random.random() < base_loss_rate:
                self.packets['lost'].add(packet_id)
            else:
                self.packets['processed'].add(packet_id)
                processed_packets.add(packet_id)
            del self.packet_timestamps[packet_id]
          
        self.traffic = len(self.packets['in_queue']) * PACKET_SIZE / 1024
        self.last_process_time = current_time
        
        return processed_packets

    def calculate_packet_loss(self, packets_to_check, current_loss_rate):
        lost_packets = set()
        for packet_id in packets_to_check:
            if random.random() < (current_loss_rate / 100.0):
                lost_packets.add(packet_id)
                self.packets['lost'].add(packet_id)
        return lost_packets

    def get_statistics(self):
        return {
            'in_transit': len(self.packets['in_queue']),
            'processed': len(self.packets['processed']),
            'dropped': len(self.packets['dropped']),
            'lost': len(self.packets['lost'])
        } 

class MEOController:
    def __init__(self, name, managed_leos):
        self.name = name
        self.managed_leos = managed_leos
        self.leo_states = {}
        self.neighbor_meo_states = {}

    def collect_leo_states(self):
        self.leo_states = {
            leo_name: {
                'traffic': leo.traffic,
                'connections': leo.connected_satellites
            }
            for leo_name, leo in self.managed_leos.items()
        }

    def exchange_states(self, other_meos):
        self.neighbor_meo_states = {}
        self.collect_leo_states()
        for meo in other_meos:
            if set(self.managed_leos.keys()) & set(meo.managed_leos.keys()):
                self.neighbor_meo_states[meo.name] = {
                    'leo_states': meo.leo_states,
                }

    def find_cross_region(self, source_leo, destination_leo):
        """在MEOController中调试find_cross_region方法"""
        print(f"查找从 {source_leo} 到 {destination_leo} 的交叉区域")
        
        # 获取源和目标所属的MEO区域
        source_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() 
                          if source_leo in info['leo_states']), None)
        dest_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() 
                        if destination_leo in info['leo_states']), None)
        
        print(f"源MEO: {source_meo}, 目标MEO: {dest_meo}")
        
        if source_meo is None or dest_meo is None:
            print("无法找到源或目标的MEO区域")
            # 如果在本地管理的LEO中
            source_local = source_leo in self.managed_leos
            dest_local = destination_leo in self.managed_leos
            print(f"源是本地LEO: {source_local}, 目标是本地LEO: {dest_local}")
            if source_local:
                return set(self.managed_leos.keys())
            return set()
        
        if source_meo == dest_meo:
            print(f"源和目标在同一MEO区域: {source_meo}")
            return set(self.neighbor_meo_states[source_meo]['leo_states'].keys())
        
        # 获取源和目标区域的LEO节点集合
        source_leos = set(self.neighbor_meo_states[source_meo]['leo_states'].keys() 
                         if source_meo else self.managed_leos.keys())
        dest_leos = set(self.neighbor_meo_states[dest_meo]['leo_states'].keys() 
                        if dest_meo else self.managed_leos.keys())
        
        print(f"源区域LEO数量: {len(source_leos)}, 目标区域LEO数量: {len(dest_leos)}")
        
        # 找到边界LEO
        boundary_leos = self._find_boundary_leos(source_leos, dest_leos)
        print(f"找到的边界LEO数量: {len(boundary_leos)}")
        
        return boundary_leos
    
    def _find_boundary_leos(self, source_leos, dest_leos):
        boundary_leos = set()
        for leo in source_leos:
            if any(self._are_neighbors(leo, dest_leo) for dest_leo in dest_leos):
                boundary_leos.add(leo)
        for leo in dest_leos:
            if any(self._are_neighbors(leo, source_leo) for source_leo in source_leos):
                boundary_leos.add(leo)
        return boundary_leos
    
    def _are_neighbors(self, leo1, leo2):
        state1 = (self.leo_states.get(leo1) or 
                 next((info['leo_states'].get(leo1) 
                      for info in self.neighbor_meo_states.values() 
                      if leo1 in info['leo_states']), None))
        state2 = (self.leo_states.get(leo2) or 
                 next((info['leo_states'].get(leo2) 
                      for info in self.neighbor_meo_states.values() 
                      if leo2 in info['leo_states']), None))
        if state1 and state2:
            return leo2 in state1.get('connections', set())
        return False

class SatelliteEnv:
    def __init__(self):
        self.reset()
        self.current_data_rate = DATA_GENERATION_RATE  # 添加这一行
        
    def reset(self):
        """重置环境状态"""
        self.meo_nodes, self.leo_nodes, self.links, self.links_dict, \
        self.leo_to_meo, self.leo_neighbors = self._setup_network()
        self.simulation_time = 0
        self.last_network_update = 0
        self.global_packet_id = 0
        
        # 初始化统计数据
        self.path_stats = {
            'sent': set(),
            'dropped': set(),
            'lost': set(),
            'received': set()
        }
        
        # 初始化可用动作缓存
        self.available_actions_cache = self._init_available_actions()
        
        # 初始化MEO控制器
        self.meo_controllers = self._init_meo_controllers()
        
        # 构建LEO网络图
        self.leo_graph = self._build_leo_graph()
        
        return self.get_state_size(), len(self.leo_nodes)

    def _setup_network(self):
        """设置网络拓扑"""
        meo_nodes = {f'meo{i}': Node(f'meo{i}') 
                    for i in range(1, NUM_ORBITS_MEO * SATS_PER_ORBIT_MEO + 1)}
        leo_nodes = {f'leo{i}': Node(f'leo{i}') 
                    for i in range(1, NUM_ORBITS_LEO * SATS_PER_ORBIT_LEO + 1)}
        links = []
        links_dict = {}
        leo_to_meo = {leo_name: f'meo{((i-1) // (SATS_PER_ORBIT_LEO * NUM_ORBITS_LEO // len(meo_nodes))) + 1}' 
                      for i, leo_name in enumerate(leo_nodes, 1)}
        leo_neighbors = defaultdict(set)
        
        # 创建LEO间链路
        for orbit in range(NUM_ORBITS_LEO):
            for pos in range(SATS_PER_ORBIT_LEO):
                current_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}'
                
                # 创建与上一个卫星的链路
                if pos > 0:
                    up_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos}'
                    self._create_leo_link(current_leo, up_leo, leo_nodes, links, links_dict, leo_neighbors)
                
                # 创建与下一个卫星的链路
                if pos == SATS_PER_ORBIT_LEO - 1:
                    down_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + 1}'
                else:
                    down_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 2}'
                self._create_leo_link(current_leo, down_leo, leo_nodes, links, links_dict, leo_neighbors)
                
                # 创建跨轨道链路
                if orbit > 0:
                    left_leo = f'leo{(orbit-1) * SATS_PER_ORBIT_LEO + pos + 1}'
                    self._create_leo_link(current_leo, left_leo, leo_nodes, links, links_dict, leo_neighbors)
                if orbit == NUM_ORBITS_LEO - 1:
                    right_leo = f'leo{pos + 1}'
                else:
                    right_leo = f'leo{(orbit+1) * SATS_PER_ORBIT_LEO + pos + 1}'
                self._create_leo_link(current_leo, right_leo, leo_nodes, links, links_dict, leo_neighbors)
        
        # 创建LEO-MEO链路
        for leo_name, meo_name in leo_to_meo.items():
            link = Link(leo_nodes[leo_name], meo_nodes[meo_name], 15, 15, 0.05)
            links.append(link)
            links_dict[(leo_name, meo_name)] = link
        
        # 创建MEO间链路
        for i in range(1, len(meo_nodes)):
            link = Link(meo_nodes[f'meo{i}'], meo_nodes[f'meo{i+1}'], 30, 8, 0.3)
            links.append(link)
            links_dict[(f'meo{i}', f'meo{i+1}')] = link
        
        # 连接最后一个MEO与第一个MEO
        link = Link(meo_nodes[f'meo{len(meo_nodes)}'], meo_nodes['meo1'], 30, 8, 0.3)
        links.append(link)
        links_dict[(f'meo{len(meo_nodes)}', 'meo1')] = link
        
        return meo_nodes, leo_nodes, links, links_dict, leo_to_meo, leo_neighbors 

    def _create_leo_link(self, leo1, leo2, leo_nodes, links, links_dict, leo_neighbors):
        """创建LEO卫星间链路"""
        link = Link(leo_nodes[leo1], leo_nodes[leo2], 20, 10, 0.1)
        links.append(link)
        links_dict[(leo1, leo2)] = link
        leo_neighbors[leo1].add(leo2)
        leo_neighbors[leo2].add(leo1)

    def _init_available_actions(self):
        """初始化可用动作缓存"""
        available_actions_cache = {}
        for leo_name in self.leo_nodes:
            available_actions = []
            for i, potential_next_leo in enumerate(self.leo_nodes):
                if ((leo_name, potential_next_leo) in self.links_dict or 
                    (potential_next_leo, leo_name) in self.links_dict):
                    available_actions.append(i)
            available_actions_cache[leo_name] = available_actions
        return available_actions_cache

    def _init_meo_controllers(self):
        """初始化MEO控制器"""
        meo_controllers = {}
        for meo_name in self.meo_nodes:
            managed_leos = {leo: self.leo_nodes[leo] 
                          for leo in self.leo_nodes 
                          if self.leo_to_meo[leo] == meo_name}
            meo_controllers[meo_name] = MEOController(meo_name, managed_leos)
        return meo_controllers

    def _build_leo_graph(self):
        """构建LEO网络图"""
        leo_graph = {}
        for leo_name in self.leo_nodes:
            leo_graph[leo_name] = set()
            for neighbor in self.leo_neighbors[leo_name]:
                leo_graph[leo_name].add(neighbor)
        return leo_graph

    def _update_network_state(self):
        """更新网络状态"""
        for link in self.links_dict.values():
            self._calculate_link_metrics(link.node1.name, link.node2.name)
    def _calculate_link_metrics(self, node1, node2):
        """计算链路性能指标"""
        link = self.links_dict.get((node1, node2)) or self.links_dict.get((node2, node1))
        if not link:
            return None

        if self.simulation_time - link.last_update_time >= UPDATE_INTERVAL:
            # 计算链路利用率
            utilization = min(1.0, link.traffic / QUEUE_CAPACITY if QUEUE_CAPACITY > 0 else 1.0)

            # 调整实际带宽
            congestion_factor = 1 - (utilization ** 2) * 0.5
            adjusted_bandwidth = link.current_bandwidth * 1e6 * congestion_factor

            # 计算延迟
            queue_size_bits = len(link.packets['in_queue']) * PACKET_SIZE * 8 * 1024

            # 计算排队延迟
            queue_delay = (queue_size_bits / adjusted_bandwidth) * 1000 if adjusted_bandwidth > 0 else float('inf')

            # 计算处理延迟
            processing_delay = 0.1 * len(link.packets['in_queue'])

            # 传播延迟
            propagation_delay = link.base_delay + queue_delay * 0.05

            # 设备负载影响
            device_load = len(link.packets['in_queue']) / link.max_packets
            processing_delay *= (1 + device_load)

            # 总延迟
            total_delay = propagation_delay + queue_delay + processing_delay

            # 添加随机抖动
            jitter = np.random.normal(0, total_delay * 0.05)
            total_delay = max(link.base_delay, total_delay + jitter)

            # 处理队列中的数据包
            processed_packets = link.process_queue(self.simulation_time)
            
            # 计算丢包
            base_loss_rate = link.base_loss * (1 + device_load)
            lost_packets = link.calculate_packet_loss(processed_packets, base_loss_rate * 100)
            
            # 计算实际丢包率
            total_processed = len(processed_packets)
            total_lost = len(lost_packets)
            actual_loss_rate = (total_lost / total_processed * 100) if total_processed > 0 else 0
            
            # 根据队列利用率调整丢包率
            utilization_factor = len(link.packets['in_queue']) / link.max_packets
            adjusted_loss_rate = actual_loss_rate * (1 + utilization_factor)
            

            # 更新链路状态
            link.current_delay = total_delay
            link.current_loss = adjusted_loss_rate / 100  # 转换为小数
            link.last_update_time = self.simulation_time
            
            return {
                'delay': total_delay,
                'bandwidth': link.current_bandwidth,
                'loss': adjusted_loss_rate,
                'last_update': link.last_update_time
            }

        return {
            'delay': link.current_delay,
            'bandwidth': link.current_bandwidth,
            'loss': link.current_loss * 100,  # 转换为百分比
            'last_update': link.last_update_time
        }
    def _calculate_shannon_capacity(self, bandwidth, snr):
        """计算香农容量"""
        snr_linear = 10 ** (snr / 10)
        capacity = (bandwidth * 1e6 * np.log2(1 + snr_linear)) / 1e6
        return capacity

    def step(self, current_leo, action, path):
        """执行一步环境交互"""
        self.simulation_time += TIME_STEP
        
        # 更新网络状态
        if self.simulation_time - self.last_network_update >= NETWORK_UPDATE_INTERVAL:
            self._update_network_state()
            self.last_network_update = self.simulation_time
        
        # 获取下一个LEO节点
        next_leo = list(self.leo_nodes.keys())[action]
        metrics = self._calculate_link_metrics(current_leo, next_leo)
        
        if not metrics:
            return None, -10, True, {'error': 'Invalid link'}
        
        # 处理数据包传输
        link = self.links_dict.get((current_leo, next_leo)) or self.links_dict.get((next_leo, current_leo))
        packets_to_send = self._generate_traffic_poisson()
        
        # 添加数据包到队列
        accepted_packets, dropped_packets, self.global_packet_id = link.add_packets(
            packets_to_send, self.global_packet_id, self.simulation_time
        )
        
        # 更新统计信息
        self.path_stats['sent'].update(accepted_packets)
        self.path_stats['dropped'].update(dropped_packets)
        
        # 处理队列
        processed_packets = link.process_queue(self.simulation_time)
        
        # 计算丢包
        lost_packets = link.calculate_packet_loss(processed_packets, metrics['loss'])
        self.path_stats['lost'].update(lost_packets)
        self.path_stats['received'].update(processed_packets - lost_packets)
        
        # 计算奖励
        reward = self._calculate_reward(next_leo, path[-1], metrics, path)
        
        # 判断是否结束
        done = len(path) >= MAX_PATH_LENGTH - 1
        
        # 准备信息字典
        info = {
            'next_leo': next_leo,
            'delay': metrics['delay'],
            'bandwidth': metrics['bandwidth'],
            'loss': metrics['loss'],
            'path_stats': self.path_stats
        }
        
        return self._get_state(next_leo), reward, done, info

    def _generate_traffic_poisson(self, time_interval=1.0):
        """生成泊松分布的流量"""
        # 使用当前测试的数据生成率
        current_rate = self.current_data_rate
        
        # 将 Gbps 转换为每秒数据包数
        bits_per_second = current_rate * 1e9  # 转换为 bps
        bits_per_packet = PACKET_SIZE * 8 * 1024  # 每个数据包的比特数
        packets_per_second = bits_per_second / bits_per_packet
        
        # 计算给定时间间隔内的平均数据包数
        lambda_packets = packets_per_second * time_interval
        
        # 使用泊松分布生成实际数据包数
        generated_packets = np.random.poisson(lambda_packets)
        
        # 添加随机变化（±10%）
        variation = random.uniform(0.9, 1.1)
        final_packets = max(1, int(generated_packets * variation))
        
        return final_packets

    def _calculate_reward(self, next_leo, destination, metrics, path):
        """计算奖励"""
        if next_leo == destination:
            return 20.0 + max(0, (MAX_PATH_LENGTH - len(path)) * 0.5)
        
        reward = -0.05  # 基础惩罚
        
        # 同区域奖励
        if self.leo_to_meo[next_leo] == self.leo_to_meo[destination]:
            reward += 1.0
            
        # 环路惩罚
        if next_leo in path:
            reward -= 0.5
            
        return reward

    def get_state_size(self):
        """获取状态空间大小"""
        return 17

    def get_available_actions(self, current_leo):
        """获取当前可用动作"""
        return self.available_actions_cache[current_leo]

    def get_leo_names(self):
        """获取LEO卫星名称列表"""
        return list(self.leo_nodes.keys())

    def get_leo_to_meo_mapping(self):
        """获取LEO到MEO的映射关系"""
        return self.leo_to_meo 

    def _get_state(self, current_leo):
        """获取当前状态向量"""
        state = []
        
        # 1. 当前节点的队列状态
        current_traffic = 0
        for link in self.links:
            if link.node1.name == current_leo or link.node2.name == current_leo:
                current_traffic += len(link.packets['in_queue'])
        state.append(current_traffic / QUEUE_CAPACITY)
        
        # 2. 邻居节点的状态
        neighbors = self.leo_neighbors[current_leo]
        neighbor_states = []
        for _ in range(8):  # 固定8个邻居位置
            if neighbors:
                neighbor = neighbors.pop()
                neighbor_traffic = 0
                for link in self.links:
                    if link.node1.name == neighbor or link.node2.name == neighbor:
                        neighbor_traffic += len(link.packets['in_queue'])
                neighbor_states.append(neighbor_traffic / QUEUE_CAPACITY)
            else:
                neighbor_states.append(0)
        state.extend(neighbor_states)
        
        # 3. 链路性能指标
        for neighbor in list(self.leo_neighbors[current_leo])[:8]:  # 最多考虑8个邻居
            metrics = self._calculate_link_metrics(current_leo, neighbor)
            if metrics:
                state.extend([
                    metrics['delay'] / 100,  # 归一化延迟
                    metrics['bandwidth'] / 20,  # 归一化带宽
                    metrics['loss'] / 100  # 归一化丢包率
                ])
            else:
                state.extend([0, 0, 0])
        
        # 填充到固定长度
        while len(state) < 17:  # 确保状态向量长度为17
            state.append(0)
        
        return state
    def _find_k_shortest_paths_with_cross_region(self, source, destination, k, graph):
        """基于最小交叉区域的k最短路径算法"""
        # 确保正确获取MEO区域
        source_meo = self.leo_to_meo[source]
        dest_meo = self.leo_to_meo[destination]
        
        print(f"\n开始寻找从 {source}({source_meo}) 到 {destination}({dest_meo}) 的路径")
        
        # 确保是跨区域路径
        if source_meo == dest_meo:
            print("源和目标在同一MEO区域，不是跨区域路径")
            return []
        
        # 更新MEO控制器状态
        for controller in self.meo_controllers.values():
            controller.collect_leo_states()
            controller.exchange_states([c for c in self.meo_controllers.values() if c != controller])
        
        # 获取交叉区域
        cross_region = set()
        for leo_name, leo_node in self.leo_nodes.items():
            if (self.leo_to_meo[leo_name] == source_meo and 
                any(self.leo_to_meo[neighbor] == dest_meo for neighbor in self.leo_neighbors[leo_name])):
                cross_region.add(leo_name)
        
        print(f"找到的交叉区域节点: {cross_region}")
        
        if not cross_region:
            print("未找到交叉区域节点")
            return []
        
        unique_paths = set()
        paths_with_costs = []
        
        def is_valid_path(path):
            """检查路径是否有效（无回路且正确跨区域）"""
            # 检查回路
            if len(path) != len(set(path)):
                return False
            
            # 检查是否正确跨区域
            crossed = False
            for i in range(len(path)-1):
                curr_meo = self.leo_to_meo[path[i]]
                next_meo = self.leo_to_meo[path[i+1]]
                if curr_meo != next_meo:
                    crossed = True
                    break
            return crossed
        
        # 对每个交叉区域节点
        for boundary_leo in cross_region:
            if boundary_leo == source or boundary_leo == destination:
                continue
            
            # 找到源到边界的路径
            source_paths = []
            excluded_nodes = set()
            
            # 为源到边界找到k条路径
            for i in range(k):
                path, cost = self._dijkstra(source, boundary_leo, excluded_nodes)
                if path is None or len(path) < 2:
                    break
                if len(set(path)) == len(path):  # 确保没有回路
                    source_paths.append((path, cost))
                excluded_nodes.update(path[1:-1])
            
            # 为边界到目标找到k条路径
            boundary_paths = []
            excluded_nodes = set()
            
            for i in range(k):
                path, cost = self._dijkstra(boundary_leo, destination, excluded_nodes)
                if path is None or len(path) < 2:
                    break
                if len(set(path)) == len(path):  # 确保没有回路
                    boundary_paths.append((path, cost))
                excluded_nodes.update(path[1:-1])
            
            # 组合路径
            for path1, cost1 in source_paths:
                for path2, cost2 in boundary_paths:
                    combined_path = path1[:-1] + path2
                    if (len(combined_path) <= MAX_PATH_LENGTH and 
                        is_valid_path(combined_path)):
                        path_tuple = tuple(combined_path)
                        if path_tuple not in unique_paths:
                            unique_paths.add(path_tuple)
                            paths_with_costs.append((combined_path, cost1 + cost2))
        
        # 按总成本排序并返回前k条不同的路径
        paths_with_costs.sort(key=lambda x: x[1])
        result_paths = []
        
        for path, _ in paths_with_costs:
            if len(result_paths) == k:
                break
            result_paths.append(path)
            print(f"找到路径: {' -> '.join(path)}")
        
        return result_paths

    def get_candidate_actions(self, current_leo, destination, available_actions):
        """获取候选动作"""
        paths = self._find_k_shortest_paths_with_cross_region(current_leo, destination, 3, self.leo_graph)
        
        candidate_actions = set()
        
        # 从候选路径中提取下一步可能的动作
        for path in paths:
            if len(path) > 1 and path[0] == current_leo:
                next_leo = path[1]
                action_idx = list(self.leo_nodes.keys()).index(next_leo)
                if action_idx in available_actions:
                    candidate_actions.add(action_idx)
        
        # 如果没有找到候选动作，返回所有可用动作
        if not candidate_actions:
            return available_actions
        
        return list(candidate_actions)  

    def _dijkstra(self, start, end, excluded_nodes=None):
        """Dijkstra最短路径算法"""
        if excluded_nodes is None:
            excluded_nodes = set()

        # 初始化距离和前驱节点字典
        distances = {node: float('infinity') for node in self.leo_graph}
        distances[start] = 0
        previous = {node: None for node in self.leo_graph}
        
        # 未访问节点集合
        unvisited = set(node for node in self.leo_graph if node not in excluded_nodes)

        while unvisited:
            # 获取未访问节点中距离最小的节点
            current = min(unvisited, key=lambda x: distances[x])
            
            # 如果到达目标节点，结束搜索
            if current == end:
                break
            
            # 从未访问集合中移除当前节点
            unvisited.remove(current)
            
            # 更新相邻节点的距离
            for neighbor in self.leo_graph[current]:
                if neighbor in excluded_nodes:
                    continue
                    
                # 计算通过当前节点到达邻居节点的距离
                distance = distances[current] + 1
                
                # 如果找到更短的路径，更新距离和前驱节点
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current

        # 如果无法到达终点，返回None
        if distances[end] == float('infinity'):
            return None, float('infinity')

        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # 返回反转后的路径和总距离
        return path[::-1], distances[end]  

    def get_cross_region_size(self, source_leo, destination_leo):
        """获取两个LEO卫星所在MEO区域之间的交叉区域大小"""
        # 获取源和目标的MEO区域
        source_meo = self.leo_to_meo[source_leo]
        dest_meo = self.leo_to_meo[destination_leo]
        
        print(f"\n计算从 {source_leo}({source_meo}) 到 {destination_leo}({dest_meo}) 的交叉区域大小")
        
        # 如果在同一MEO区域，返回该区域的所有LEO数量
        if source_meo == dest_meo:
            region_size = sum(1 for leo in self.leo_to_meo if self.leo_to_meo[leo] == source_meo)
            print(f"源和目标在同一MEO区域，区域大小: {region_size}")
            return region_size
        
        # 找到交叉区域的LEO节点
        cross_region = set()
        for leo_name in self.leo_nodes:
            # 如果LEO在源MEO区域
            if self.leo_to_meo[leo_name] == source_meo:
                # 检查是否与目标MEO区域的任何LEO相邻
                for neighbor in self.leo_neighbors[leo_name]:
                    if self.leo_to_meo[neighbor] == dest_meo:
                        cross_region.add(leo_name)
                        cross_region.add(neighbor)
        
        print(f"交叉区域节点数量: {len(cross_region)}")
        print(f"交叉区域节点: {cross_region}")
        
        return len(cross_region)  