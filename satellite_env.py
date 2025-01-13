from config import *
from collections import defaultdict, deque
from datetime import datetime
import math
import time
import random
import numpy as np

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
       
        # 获取源和目标所属的MEO区域
        source_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() 
                          if source_leo in info['leo_states']), None)
        dest_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() 
                        if destination_leo in info['leo_states']), None)
        
        
        if source_meo is None or dest_meo is None:
            # 如果在本地管理的LEO中
            source_local = source_leo in self.managed_leos
            dest_local = destination_leo in self.managed_leos
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
        
        
        # 找到边界LEO
        boundary_leos = self._find_boundary_leos(source_leos, dest_leos)
        
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
        current_time = self.simulation_time
        
        # 更新所有链路的状态
        for link in self.links:
            # 只在距离上次更新超过间隔时更新
            if current_time - link.last_update_time >= NETWORK_UPDATE_INTERVAL:
                # 处理队列中的数据包
                link.process_queue(current_time)
                
                # 更新链路性能指标
                metrics = self._calculate_link_metrics(link.node1.name, link.node2.name)
                if metrics:
                    # 更新链路参数
                    link.current_delay = metrics['delay']
                    link.current_bandwidth = metrics['bandwidth']
                    link.current_loss = metrics['loss'] / 100  # 转换回小数
                    
                    # 更新节点流量
                    link.node1.traffic = len(link.packets['in_queue']) * PACKET_SIZE / 1024
                    link.node2.traffic = link.node1.traffic
                    
                # 更新最后更新时间
                link.last_update_time = current_time
                
        # 更新MEO控制器状态
        for controller in self.meo_controllers.values():
            controller.collect_leo_states()

    def _calculate_link_metrics(self, source, destination):
        """计算链路性能指标"""
        link = self.links_dict.get((source, destination)) or \
               self.links_dict.get((destination, source))
        
        if not link:
            return None
        
        # 计算实际卫星间距离
        try:
            distance = self._calculate_satellite_distance(source, destination)
        except ValueError:
            # 如果涉及MEO卫星，使用默认距离
            distance = 1000  # 默认距离为1000km
        
        # 1. 延迟计算优化
        # 1.1 传播延迟 (光速传播)
        propagation_delay = (distance / 300000 * 1000) * 0.8  # 降低传播延迟的影响
        
        # 1.2 排队和传输延迟计算优化
        queue_size = len(link.packets['in_queue'])
        if queue_size > 0:
            # 计算单个数据包的传输时间 (优化参数)
            packet_bits = PACKET_SIZE * 8 * 1024  # bits
            available_bandwidth = max(1, link.current_bandwidth) * 1e6  # bps
            transmission_time = (packet_bits / available_bandwidth * 1000) * 0.5  # 降低基础传输时间
            
            # 优化排队延迟计算
            queue_ratio = queue_size / link.max_packets
            
            # 增加并行处理能力
            parallel_capacity = max(1, int(link.current_bandwidth * 4))  # 进一步增加并行处理能力
            effective_queue_size = max(1, queue_size / parallel_capacity)
            
            # 大幅降低排队延迟
            if queue_ratio <= 0.3:  # 轻载
                queuing_delay = transmission_time * (effective_queue_size / 16)  # 进一步降低系数
            elif queue_ratio <= 0.7:  # 中等负载
                queuing_delay = transmission_time * (effective_queue_size / 8)  # 进一步降低系数
            else:  # 重载
                congestion_factor = 1 + (queue_ratio - 0.7) * 0.3  # 进一步降低拥塞影响
                queuing_delay = transmission_time * (effective_queue_size / 4) * congestion_factor
            
            # 优化传输延迟
            transmission_delay = transmission_time * (1 + queue_ratio * 0.05)  # 进一步降低队列影响
            
        else:
            queuing_delay = 0
            transmission_delay = 0
        
        # 总延迟优化
        processing_overhead = 0.02  # 进一步降低处理开销
        total_delay = (propagation_delay * 0.8 +  # 降低传播延迟权重
                      queuing_delay * 0.2 +  # 进一步降低排队延迟权重
                      transmission_delay * 0.3 +  # 进一步降低传输延迟权重
                      processing_overhead)
        
        
        # 2. 带宽计算
        link_utilization = link.traffic / QUEUE_CAPACITY
        congestion_factor = math.tanh(link_utilization)
        effective_bandwidth = link.base_bandwidth * (1 - congestion_factor * 0.4)
        
        # 3. 丢包率计算
        if link.max_packets > 0:
            queue_drop_rate = len(link.packets['dropped']) / max(1, (len(link.packets['in_queue']) + 
                                                                   len(link.packets['dropped'])))
        else:
            queue_drop_rate = 0
        
        base_loss_rate = link.base_loss * (1 + link_utilization)
        weather_factor = link.weather_factor
        transmission_loss_rate = base_loss_rate * weather_factor
        total_loss_rate = queue_drop_rate + transmission_loss_rate - (queue_drop_rate * transmission_loss_rate)
        
        return {
            'delay': total_delay,
            'bandwidth': effective_bandwidth,
            'loss': total_loss_rate * 100,
            'utilization': link_utilization * 100,
            'queue_delay': queuing_delay,
            'transmission_delay': transmission_delay,
            'propagation_delay': propagation_delay,
            'queue_size': queue_size,
            'max_queue': link.max_packets
        }

    def _calculate_satellite_distance(self, sat1, sat2):
        """计算两颗卫星之间的距离(km)"""
        # 检查是否涉及MEO卫星
        if sat1.startswith('meo') or sat2.startswith('meo'):
            return 1000  # MEO-LEO或MEO-MEO链路的默认距离
        
        # 从卫星名称中提取轨道和位置信息
        orbit1, pos1 = self._get_orbit_position(sat1)
        orbit2, pos2 = self._get_orbit_position(sat2)
        
        # LEO卫星轨道参数
        altitude = 1000  # LEO轨道高度(km)
        earth_radius = 6371  # 地球半径(km)
        orbit_radius = earth_radius + altitude
        sats_per_orbit = SATS_PER_ORBIT_LEO
        num_orbits = NUM_ORBITS_LEO
        
        # 计算卫星在轨道平面中的角度
        angle_in_orbit1 = 2 * math.pi * pos1 / sats_per_orbit
        angle_in_orbit2 = 2 * math.pi * pos2 / sats_per_orbit
        
        # 计算轨道平面间的角度
        orbit_angle = 2 * math.pi * (orbit1 - orbit2) / num_orbits
        
        # 计算两颗卫星的笛卡尔坐标
        x1 = orbit_radius * math.cos(angle_in_orbit1)
        y1 = orbit_radius * math.sin(angle_in_orbit1) * math.cos(orbit_angle)
        z1 = orbit_radius * math.sin(angle_in_orbit1) * math.sin(orbit_angle)
        
        x2 = orbit_radius * math.cos(angle_in_orbit2)
        y2 = orbit_radius * math.sin(angle_in_orbit2)
        z2 = 0  # 参考轨道平面
        
        # 计算直线距离
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        return distance

    def _get_orbit_position(self, sat_name):
        """从卫星名称中提取轨道编号和位置编号"""
        if not sat_name.startswith('leo'):
            raise ValueError(f"Invalid satellite name: {sat_name}. Only LEO satellites are supported.")
        
        try:
            # 提取LEO编号
            sat_num = int(sat_name.replace('leo', ''))
            orbit_num = (sat_num - 1) // SATS_PER_ORBIT_LEO
            position = (sat_num - 1) % SATS_PER_ORBIT_LEO
            return orbit_num, position
        except ValueError as e:
            raise ValueError(f"Error parsing satellite name {sat_name}: {str(e)}")

    def _calculate_shannon_capacity(self, bandwidth, snr):
        """计算香农容量
        
        Args:
            bandwidth: 带宽 (MHz)
            snr: 信噪比 (dB)
        
        Returns:
            float: 理论最大带宽 (MHz)
        """
        snr_linear = 10 ** (snr / 10)  # 转换为线性值
        capacity = bandwidth * math.log2(1 + snr_linear)  # MHz
        return capacity

    def step(self, current_leo, action, path):
        """执行一步环境交互"""
        self.simulation_time += TIME_STEP
        
        # 获取下一个LEO节点
        next_leo = list(self.leo_nodes.keys())[action]
        
        # 获取链路
        link = self.links_dict.get((current_leo, next_leo)) or self.links_dict.get((next_leo, current_leo))
        if not link:
            return None, -10, True, {'error': 'Invalid link'}
        
        # 更新网络状态
        if self.simulation_time - self.last_network_update >= NETWORK_UPDATE_INTERVAL:
            self._update_network_state()
            self.last_network_update = self.simulation_time
        
        # 处理数据包传输
        packets_to_send = self._generate_traffic_poisson()
        accepted_packets, dropped_packets, self.global_packet_id = link.add_packets(
            packets_to_send, self.global_packet_id, self.simulation_time
        )
        
        # 处理队列
        processed_packets = link.process_queue(self.simulation_time)
        
        # 计算链路指标
        metrics = self._calculate_link_metrics(current_leo, next_leo)
        if not metrics:
            metrics = {
                'delay': link.base_delay,
                'bandwidth': link.base_bandwidth,
                'loss': link.base_loss
            }
        
        # 计算丢包
        lost_packets = link.calculate_packet_loss(processed_packets, metrics['loss'])
        
        # 更新统计信息
        self.path_stats['sent'].update(accepted_packets)
        self.path_stats['dropped'].update(dropped_packets)
        self.path_stats['lost'].update(lost_packets)
        self.path_stats['received'].update(processed_packets - lost_packets)
        
        # 计算队列利用率
        queue_utilization = len(link.packets['in_queue']) / link.max_packets if link.max_packets > 0 else 0
        bandwidth_utilization = link.traffic / QUEUE_CAPACITY if QUEUE_CAPACITY > 0 else 0
        
        # 计算丢包率 (修改这部分)
        total_packets = len(accepted_packets) + len(dropped_packets)
        if total_packets > 0:
            drop_rate = (len(dropped_packets) + len(lost_packets)) / total_packets * 100
        else:
            drop_rate = 0.0
        
        # 准备性能指标
        link_stats = {
            'delay': metrics['delay'],
            'bandwidth': metrics['bandwidth'],
            'loss': drop_rate,  # 使用新计算的丢包率
            'queue_utilization': queue_utilization * 100,
            'bandwidth_utilization': bandwidth_utilization * 100,
            'packets_in_queue': len(link.packets['in_queue']),
            'packets_processed': len(processed_packets),
            'packets_dropped': len(dropped_packets),
            'packets_lost': len(lost_packets)
        }
        
        # 计算奖励
        reward = self._calculate_reward(next_leo, path[-1], metrics, path)
        
        # 判断是否结束
        done = len(path) >= MAX_PATH_LENGTH - 1
        
        # 返回信息
        info = {
            'next_leo': next_leo,
            'link_stats': link_stats,
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
        return 18

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
        
        # 4. 添加一个额外的特征（例如：当前时间归一化）
        state.append(self.simulation_time / (24 * 60 * 60))  # 假设一天为周期
        
        # 填充到固定长度
        while len(state) < 18:  # 确保状态向量长度为18
            state.append(0)
        
        return state[:18]  # 确保返回18维向量
    def _find_k_shortest_paths_with_cross_region(self, source, destination, k, graph):
        """基于最小交叉区域的k最短路径算法"""
        # 获取源和目标的MEO区域
        source_meo = self.leo_to_meo[source]
        dest_meo = self.leo_to_meo[destination]
        
        # 如果在同一MEO区域，使用改进的k最短路径算法
        if source_meo == dest_meo:
            paths = []
            excluded_edges = set()  # 使用边而不是节点来避免
            for _ in range(k):
                # 使用当前的excluded_edges找到最短路径
                path = self._modified_dijkstra(source, destination, excluded_edges)
                if not path or len(path) < 2:
                    break
                paths.append(path)
                
                # 从当前路径中随机选择一条边加入到excluded_edges
                for i in range(len(path)-1):
                    excluded_edges.add((path[i], path[i+1]))
                    excluded_edges.add((path[i+1], path[i]))  # 双向都要排除
                
            return paths

        # 找到交叉区域节点
        cross_region = set()
        for leo1, neighbors in self.leo_neighbors.items():
            meo1 = self.leo_to_meo[leo1]
            for leo2 in neighbors:
                meo2 = self.leo_to_meo[leo2]
                if ((meo1 == source_meo and meo2 == dest_meo) or 
                    (meo1 == dest_meo and meo2 == source_meo)):
                    cross_region.add(leo1)
                    cross_region.add(leo2)
        
        if not cross_region:
            # 如果没有直接的交叉区域，尝试通过中间MEO区域寻找路径
            all_meos = set(self.leo_to_meo.values())
            intermediate_meos = all_meos - {source_meo, dest_meo}
            
            for inter_meo in intermediate_meos:
                source_boundary = self._find_boundary_between_meos(source_meo, inter_meo)
                dest_boundary = self._find_boundary_between_meos(inter_meo, dest_meo)
                
                if source_boundary and dest_boundary:
                    cross_region.update(source_boundary)
                    cross_region.update(dest_boundary)
        
        if not cross_region:
            return []
        
        # 使用交叉区域节点构建多条不同的路径
        paths = []
        used_paths = set()  # 用于跟踪已使用的路径
        
        for boundary_leo in cross_region:
            excluded_edges = set()
            
            for _ in range(k):
                # 寻找源到边界的路径
                path1 = self._modified_dijkstra(source, boundary_leo, excluded_edges)
                if not path1:
                    continue
                    
                # 寻找边界到目标的路径
                path2 = self._modified_dijkstra(boundary_leo, destination, excluded_edges)
                if not path2:
                    continue
                
                # 组合完整路径
                complete_path = path1[:-1] + path2
                path_key = tuple(complete_path)  # 转换为tuple以便用作set的元素
                
                # 检查路径是否已存在且长度是否合适
                if path_key not in used_paths and len(complete_path) <= MAX_PATH_LENGTH:
                    paths.append(complete_path)
                    used_paths.add(path_key)
                    
                    # 随机选择一条边加入到excluded_edges
                    if len(complete_path) > 1:
                        i = random.randint(0, len(complete_path)-2)
                        excluded_edges.add((complete_path[i], complete_path[i+1]))
                        excluded_edges.add((complete_path[i+1], complete_path[i]))
            
        return paths[:k]

    def _modified_dijkstra(self, source, target, excluded_edges):
        """修改后的Dijkstra算法，考虑被排除的边"""
        distances = {node: float('infinity') for node in self.leo_nodes}
        distances[source] = 0
        predecessors = {node: None for node in self.leo_nodes}
        unvisited = set(self.leo_nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            
            if current == target:
                break
            
            unvisited.remove(current)
            
            if distances[current] == float('infinity'):
                break
            
            # 检查邻居节点，排除被禁用的边
            for neighbor in self.leo_neighbors[current]:
                if (current, neighbor) in excluded_edges:
                    continue
                    
                metrics = self._calculate_link_metrics(current, neighbor)
                if not metrics:
                    continue
                    
                distance = distances[current] + metrics['delay']
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
        
        # 构建路径
        if target not in predecessors or predecessors[target] is None:
            return None
        
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()
        return path

    def _find_boundary_between_meos(self, meo1, meo2):
        """找到两个MEO区域之间的边界节点"""
        boundary = set()
        for leo1, neighbors in self.leo_neighbors.items():
            if self.leo_to_meo[leo1] == meo1:
                for leo2 in neighbors:
                    if self.leo_to_meo[leo2] == meo2:
                        boundary.add(leo1)
                        boundary.add(leo2)
        return boundary

    def get_cross_region_size(self, source_leo, destination_leo):
        """获取两个LEO卫星所在MEO区域之间的交叉区域大小"""
        source_meo = self.leo_to_meo[source_leo]
        dest_meo = self.leo_to_meo[destination_leo]
        
        
        # 如果在同一MEO区域，返回该区域的所有LEO数量
        if source_meo == dest_meo:
            region_size = sum(1 for leo in self.leo_to_meo if self.leo_to_meo[leo] == source_meo)
            return region_size
        
        # 找到直接交叉区域
        cross_region = set()
        for leo1, neighbors in self.leo_neighbors.items():
            meo1 = self.leo_to_meo[leo1]
            for leo2 in neighbors:
                meo2 = self.leo_to_meo[leo2]
                if ((meo1 == source_meo and meo2 == dest_meo) or 
                    (meo1 == dest_meo and meo2 == source_meo)):
                    cross_region.add(leo1)
                    cross_region.add(leo2)
        
        # 如果没有直接交叉，寻找通过中间MEO的路径
        if not cross_region:
            all_meos = set(self.leo_to_meo.values())
            intermediate_meos = all_meos - {source_meo, dest_meo}
            
            for inter_meo in intermediate_meos:
                source_boundary = self._find_boundary_between_meos(source_meo, inter_meo)
                dest_boundary = self._find_boundary_between_meos(inter_meo, dest_meo)
                
                if source_boundary and dest_boundary:
                    cross_region.update(source_boundary)
                    cross_region.update(dest_boundary)
       
        
        return len(cross_region)  

    def _dijkstra(self, source, target, excluded_nodes=None):
        """使用Dijkstra算法找到最短路径
        
        Args:
            source: 源节点
            target: 目标节点
            excluded_nodes: 需要排除的节点集合
        
        Returns:
            tuple: (path, distance) 路径和总距离
        """
        if excluded_nodes is None:
            excluded_nodes = set()
        
        # 初始化距离和前驱节点字典
        distances = {node: float('infinity') for node in self.leo_nodes}
        distances[source] = 0
        predecessors = {node: None for node in self.leo_nodes}
        
        # 未访问节点集合
        unvisited = set(node for node in self.leo_nodes if node not in excluded_nodes)
        
        while unvisited:
            # 找到未访问节点中距离最小的节点
            current = min(unvisited, key=lambda x: distances[x])
            
            if current == target:
                break
            
            unvisited.remove(current)
            
            # 如果当前节点的距离是无穷大，说明无法到达目标
            if distances[current] == float('infinity'):
                break
            
            # 更新邻居节点的距离
            for neighbor in self.leo_neighbors[current]:
                if neighbor in excluded_nodes:
                    continue
                
                # 获取链路性能指标
                metrics = self._calculate_link_metrics(current, neighbor)
                if not metrics:
                    continue
                
                distance = distances[current] 
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
        
        # 构建路径
        if target not in predecessors or predecessors[target] is None:
            return None, float('infinity')
        
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()
        
        return path, distances[target]  

    def get_candidate_actions(self, current_leo, destination, available_actions):
        """获取基于当前状态的候选动作
        
        Args:
            current_leo: 当前LEO卫星
            destination: 目标LEO卫星
            available_actions: 可用动作列表
        
        Returns:
            list: 候选动作列表
        """
        # 获取所有可能的路径
        paths = self._find_k_shortest_paths_with_cross_region(current_leo, destination, 3, self.leo_graph)
        candidate_actions = set()
        
        # 从路径中提取下一步可能的动作
        for path in paths:
            if path and len(path) > 1:
                next_leo = path[1]  # 获取路径中的下一个节点
                # 将节点名称转换为动作索引
                action_idx = list(self.leo_nodes.keys()).index(next_leo)
                if action_idx in available_actions:
                    candidate_actions.add(action_idx)
        
        # 如果没有找到候选动作，返回所有可用动作
        if not candidate_actions:
            return available_actions
        
        return list(candidate_actions)  