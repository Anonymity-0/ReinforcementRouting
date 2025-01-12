import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math

# 全局常量 (使用元组替代列表以节省内存)
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
QUEUE_CAPACITY = 200  # MB
PACKET_SIZE = 25  # KB

# 简化数据生成率相关的全局常量
DATA_GENERATION_RATE = 5.0  # 固定数据生成率（Gbps）

# 添加新的常量
BANDWIDTH = 20  # 基础带宽 (MHz)
SNR_MIN = 10  # 最小信噪比 (dB)
SNR_MAX = 30  # 最大信噪比 (dB)

UPDATE_INTERVAL = 100  # 网络状态更新间隔（毫秒）

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, leo_names, leo_to_meo):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.leo_names = leo_names
        self.leo_to_meo = leo_to_meo
        
        # 添加学习参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = INITIAL_EPSILON
        self.epsilon_min = MIN_EPSILON
        self.epsilon_decay = DECAY_RATE
        self.learning_rate = 0.001  # 添加学习率
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_state(self, current_leo, destination_leo, links_dict, leo_nodes):
        # 获取当前LEO的性能指标
        current_metrics = []
        for neighbor in leo_nodes[current_leo].connected_satellites:
            metrics = calculate_link_metrics(current_leo, neighbor, links_dict)
            if metrics:
                current_metrics.extend([
                    metrics['delay'] / 100.0,  # 归一化
                    metrics['bandwidth'] / 20.0,
                    metrics['loss'],
                    links_dict.get((current_leo, neighbor)).traffic / QUEUE_CAPACITY
                ])
        
        # 添加目标LEO的位置信息
        dest_meo = self.leo_to_meo[destination_leo]
        current_meo = self.leo_to_meo[current_leo]
        same_region = 1.0 if dest_meo == current_meo else 0.0
        
        # 组合状态向量
        state = current_metrics + [same_region]
        
        # 填充固定长度
        max_neighbors = 4  # 假设最多4个邻居
        expected_length = max_neighbors * 4 + 1  # 4个指标 + 1个区域标识
        if len(state) < expected_length:
            state.extend([0.0] * (expected_length - len(state)))
        
        return torch.FloatTensor(state).to(self.device)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_candidate_actions(self, current_leo, destination, meo_controllers, links_dict, leo_graph, available_actions):
        """获取候选动作"""
        # 使用find_best_path获取候选路径
        candidate_paths = find_k_shortest_paths(current_leo, destination, 3, leo_graph)
        candidate_actions = set()
        
        # 从候选路径中提取下一步可能的动作
        for path in candidate_paths:
            if len(path) > 1 and path[0] == current_leo:
                next_leo = path[1]
                action_idx = self.leo_names.index(next_leo)
                if action_idx in available_actions:
                    candidate_actions.add(action_idx)
        
        # 如果没有找到候选动作，返回所有可用动作
        if not candidate_actions:
            return available_actions
        
        return list(candidate_actions)

    def calculate_path_quality(self, metrics):
        """计算路径质量得分"""
        if not metrics:
            return float('-inf')
            
        delay_score = max(0, 1 - metrics['delay'] / 200)  # 延迟越低越好
        bandwidth_score = metrics['bandwidth'] / 20  # 带宽越高越好
        loss_score = max(0, 1 - metrics['loss'])  # 丢包率越低越好
        
        # 综合得分，权重可调
        return (delay_score * 0.4 + bandwidth_score * 0.4 + loss_score * 0.2)

    def choose_action(self, state, available_actions, current_leo, destination, meo_controllers, links_dict, leo_graph, path):
        """改进的动作选择策略，增强环路预防"""
        if len(available_actions) == 0:
            return None
            
        # 移除会导致环路的动作
        non_loop_actions = [a for a in available_actions if self.leo_names[a] not in path]
        
        # 如果所有动作都会导致环路，返回None
        if not non_loop_actions:
            return None
            
        # 获取候选动作
        candidate_actions = self.get_candidate_actions(
            current_leo, destination, meo_controllers, links_dict, leo_graph, non_loop_actions
        )
        
        # 探索
        if random.random() < self.epsilon:
            if random.random() < 0.8 and candidate_actions:
                # 根据路径质量选择候选动作
                best_action = None
                best_score = float('-inf')
                
                for action in candidate_actions:
                    next_leo = self.leo_names[action]
                    metrics = calculate_link_metrics(current_leo, next_leo, links_dict)
                    score = self.calculate_path_quality(metrics)
                    
                    # 增加目标导向的评分
                    if self.leo_to_meo[next_leo] == self.leo_to_meo[destination]:
                        score *= 1.2  # 增加同区域卫星的优先级
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                if best_action is not None:
                    return best_action
                    
            return random.choice(non_loop_actions)
        
        # 利用
        with torch.no_grad():
            state = state.unsqueeze(0)
            action_values = self.policy_net(state)
            
            # 计算每个动作的综合得分
            action_scores = []
            for action in non_loop_actions:
                next_leo = self.leo_names[action]
                metrics = calculate_link_metrics(current_leo, next_leo, links_dict)
                path_quality = self.calculate_path_quality(metrics)
                q_value = action_values[0][action].item()
                
                # 综合考虑Q值、路径质量和目标导向
                score = 0.5 * q_value + 0.3 * path_quality
                
                # 增加目标导向的权重
                if self.leo_to_meo[next_leo] == self.leo_to_meo[destination]:
                    score *= 1.2
                
                action_scores.append((action, score))
            
            # 选择得分最高的动作
            return max(action_scores, key=lambda x: x[1])[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([s[0] for s in batch])
        actions = torch.tensor([s[1] for s in batch], device=self.device)
        rewards = torch.tensor([s[2] for s in batch], device=self.device, dtype=torch.float32)
        next_states = torch.stack([s[3] for s in batch])
        dones = torch.tensor([s[4] for s in batch], device=self.device, dtype=torch.float32)

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # 计算损失并更新
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _calculate_bandwidth(self, base_bw, utilization, weather_factor):
        """计算当前带宽"""
        bandwidth_degradation = 1.0 - (utilization ** 1.5) * 0.5  # 增加利用率对带宽的影响
        dynamic_factor = 1.0 + 0.2 * np.random.normal()  # 添加随机波动
        return max(2.0, base_bw * bandwidth_degradation * weather_factor * dynamic_factor)

    def calculate_loss_components(self, utilization, base_loss, weather_factor, current_bandwidth, base_bandwidth, prob_loss):
        """计算丢包率的各个组成部分"""
        congestion_loss = utilization ** 2.5  # 降低指数，使拥塞更敏感
        weather_loss = base_loss * (3 - weather_factor * 2)  # 增加天气影响
        bandwidth_loss = max(0, 1 - (current_bandwidth / base_bandwidth)) * 0.2
        time_based_loss = 0.05 * np.sin(time.time() / 3600)  # 添加时间相关波动
        return min(1.0, congestion_loss + weather_loss + bandwidth_loss + prob_loss + time_based_loss)

    def get_base_parameters(self, node1, node2):
        """获取基础参数设置"""
        # 原代码
        self.base_params = {
            'meo_to_meo': (30, 8, 0.3),
            'leo_to_meo': (15, 15, 0.05),
            'leo_to_leo': (20, 10, 0.1)
        }
        
        # 建议修改为
        self.base_params = {
            'meo_to_meo': (30, random.uniform(6, 12), random.uniform(0.2, 0.4)),
            'leo_to_meo': (15, random.uniform(12, 18), random.uniform(0.03, 0.08)),
            'leo_to_leo': (20, random.uniform(8, 15), random.uniform(0.05, 0.15))
        }

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
        
        # 添加时间相关属性
        self.last_process_time = 0
        self.last_update_time = 0
        
        # 计算队列容量（以包数为单位）
        self.max_packets = int((QUEUE_CAPACITY * 1024 * 1024) / (PACKET_SIZE * 1024))  # 将MB转换为KB
        
        # 数据包追踪
        self.packets = {
            'in_queue': set(),      # 当前在队列中的包
            'processed': set(),      # 已处理的包
            'dropped': set(),        # 因队列满被丢弃的包
            'lost': set()           # 传输过程中丢失的包
        }
        self.packet_timestamps = {}  # 记录包的到达时间

    def add_packets(self, num_packets, start_id, current_time):
        """添加数据包到队列,超出队列容量的包会被丢弃"""
        accepted_packets = set()
        dropped_packets = set()
        
        for i in range(num_packets):
            packet_id = start_id + i
            
            # 检查队列容量
            if len(self.packets['in_queue']) < self.max_packets:
                self.packets['in_queue'].add(packet_id)
                self.packet_timestamps[packet_id] = current_time
                accepted_packets.add(packet_id)
            else:
                # 队列满时直接丢弃数据包
                self.packets['dropped'].add(packet_id)
                dropped_packets.add(packet_id)
        
        # 更新链路负载
        self.traffic = len(self.packets['in_queue']) * PACKET_SIZE / 1024
        
        # 当队列接近满载时,更新拥塞状态
        self.congestion_level = len(self.packets['in_queue']) / self.max_packets
        
        return accepted_packets, dropped_packets, start_id + num_packets

    def process_queue(self, current_time):
        """处理队列中的数据包,考虑带宽限制"""
        if not self.packets['in_queue']:
            return set()
        
        time_delta = current_time - self.last_process_time
        if time_delta <= 0:
            return set()
        
        # 计算当前实际可用带宽
        available_bandwidth = self.current_bandwidth * 1e6  # 转换为bps
        
        # 计算在给定时间间隔内理论上可以处理的数据量(bits)
        processable_bits = available_bandwidth * (time_delta / 1000.0)
        packet_bits = PACKET_SIZE * 8 * 1024  # 包大小(bits)
        
        # 计算可以处理的包数量
        packets_can_process = int(processable_bits / packet_bits)
        
        # 处理队列中的数据包
        processed_packets = set()
        packets_to_process = sorted(
            [(pid, self.packet_timestamps[pid]) for pid in self.packets['in_queue']],
            key=lambda x: x[1]
        )[:packets_can_process]  # 只处理带宽允许的包数量
        
        for packet_id, _ in packets_to_process:
            self.packets['in_queue'].remove(packet_id)
            self.packets['processed'].add(packet_id)
            processed_packets.add(packet_id)
            del self.packet_timestamps[packet_id]
        
        # 更新链路状态
        self.traffic = len(self.packets['in_queue']) * PACKET_SIZE / 1024
        self.last_process_time = current_time
        
        return processed_packets

    def calculate_packet_loss(self, packets_to_check, current_loss_rate):
        """计算传输过程中丢失的包"""
        lost_packets = set()
        for packet_id in packets_to_check:
            # 使用随机数确定包是否丢失，避免取整带来的误差
            if random.random() < (current_loss_rate / 100.0):
                lost_packets.add(packet_id)
                self.packets['lost'].add(packet_id)
        
        return lost_packets

    def get_statistics(self):
        """获取链路的统计信息"""
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
    
    def find_cross_region(self, source_leo, dest_leo):
        source_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() if source_leo in info['leo_states']), None)
        dest_meo = next((meo_name for meo_name, info in self.neighbor_meo_states.items() if dest_leo in info['leo_states']), None)
        
        if source_meo is None or dest_meo is None: return set()
        
        if source_meo == dest_meo:
            return set((self.neighbor_meo_states[source_meo]['leo_states'].keys() if source_meo else self.managed_leos.keys()))
        
        source_leos = set(self.neighbor_meo_states[source_meo]['leo_states'].keys() if source_meo else self.managed_leos.keys())
        dest_leos = set(self.neighbor_meo_states[dest_meo]['leo_states'].keys() if dest_meo else self.managed_leos.keys())
        
        return self._find_boundary_leos(source_leos, dest_leos)
    
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
        state1 = (self.leo_states.get(leo1) or next((info['leo_states'].get(leo1) for info in self.neighbor_meo_states.values() if leo1 in info['leo_states']), None))
        state2 = (self.leo_states.get(leo2) or next((info['leo_states'].get(leo2) for info in self.neighbor_meo_states.values() if leo2 in info['leo_states']), None))
        if state1 and state2:
            return leo2 in state1.get('connections', set())
        return False

    def participate_global_routing(self, other_meos, source_leo, destination_leo):
        self.exchange_states(other_meos)
        cross_region = self.find_cross_region(source_leo, destination_leo)
        possible_paths = []

        if source_leo in self.managed_leos and destination_leo in self.managed_leos:
            possible_paths.append([source_leo, destination_leo])
            common_leos = set(self.managed_leos.keys()) - {source_leo, destination_leo}
            for leo in common_leos:
                if self._are_neighbors(source_leo, leo) and self._are_neighbors(leo, destination_leo):
                    possible_paths.append([source_leo, leo, destination_leo])
        else:
            for src_boundary in cross_region:
                if self._are_neighbors(source_leo, src_boundary):
                    for dst_boundary in cross_region:
                        if self._are_neighbors(dst_boundary, destination_leo) and src_boundary != dst_boundary:
                            possible_paths.append([source_leo, src_boundary, dst_boundary, destination_leo])
        return possible_paths

def setup_network():
    meo_nodes = {f'meo{i}': Node(f'meo{i}') for i in range(1, NUM_ORBITS_MEO * SATS_PER_ORBIT_MEO + 1)}
    leo_nodes = {f'leo{i}': Node(f'leo{i}') for i in range(1, NUM_ORBITS_LEO * SATS_PER_ORBIT_LEO + 1)}
    links = []
    links_dict = {}
    leo_to_meo = {leo_name: f'meo{((i-1) // (SATS_PER_ORBIT_LEO * NUM_ORBITS_LEO // len(meo_nodes))) + 1}' for i, leo_name in enumerate(leo_nodes, 1)}
    leo_neighbors = defaultdict(set)
    
    # 创建LEO间链路
    for orbit in range(NUM_ORBITS_LEO):
        for pos in range(SATS_PER_ORBIT_LEO):
            current_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 1}'
            if pos > 0:
                up_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos}'
                links.append(Link(leo_nodes[current_leo], leo_nodes[up_leo], 20, 10, 0.1))
                links_dict[(current_leo, up_leo)] = links[-1]
                leo_neighbors[current_leo].add(up_leo)
                leo_neighbors[up_leo].add(current_leo)
            if pos == SATS_PER_ORBIT_LEO - 1:
                down_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + 1}'
            else:
                down_leo = f'leo{orbit * SATS_PER_ORBIT_LEO + pos + 2}'
            links.append(Link(leo_nodes[current_leo], leo_nodes[down_leo], 20, 10, 0.1))
            links_dict[(current_leo, down_leo)] = links[-1]
            leo_neighbors[current_leo].add(down_leo)
            leo_neighbors[down_leo].add(current_leo)
            if orbit > 0:
                left_leo = f'leo{(orbit-1) * SATS_PER_ORBIT_LEO + pos + 1}'
                links.append(Link(leo_nodes[current_leo], leo_nodes[left_leo], 20, 10, 0.1))
                links_dict[(current_leo, left_leo)] = links[-1]
                leo_neighbors[current_leo].add(left_leo)
                leo_neighbors[left_leo].add(current_leo)
            if orbit == NUM_ORBITS_LEO - 1:
                right_leo = f'leo{pos + 1}'
            else:
                right_leo = f'leo{(orbit+1) * SATS_PER_ORBIT_LEO + pos + 1}'
            links.append(Link(leo_nodes[current_leo], leo_nodes[right_leo], 20, 10, 0.1))
            links_dict[(current_leo, right_leo)] = links[-1]
            leo_neighbors[current_leo].add(right_leo)
            leo_neighbors[right_leo].add(current_leo)
    
    # 创建LEO-MEO链路
    for leo_name, meo_name in leo_to_meo.items():
        links.append(Link(leo_nodes[leo_name], meo_nodes[meo_name], 15, 15, 0.05))
        links_dict[(leo_name, meo_name)] = links[-1]

    # 创建MEO间链路
    for i in range(1, len(meo_nodes)):
        links.append(Link(meo_nodes[f'meo{i}'], meo_nodes[f'meo{i+1}'], 30, 8, 0.3))
        links_dict[(f'meo{i}', f'meo{i+1}')] = links[-1]
    links.append(Link(meo_nodes[f'meo{len(meo_nodes)}'], meo_nodes['meo1'], 30, 8, 0.3))
    links_dict[(f'meo{len(meo_nodes)}', 'meo1')] = links[-1]
    
    # 优化链路创建时的基础参数
    for link in links:
        if isinstance(link, Link):
            # 增加初始带宽和降低初始丢包率
            if 'meo' in link.node1.name and 'meo' in link.node2.name:
                link.base_bandwidth *= 1.5
                link.base_loss *= 0.5
            elif 'meo' in link.node1.name or 'meo' in link.node2.name:
                link.base_bandwidth *= 1.3
                link.base_loss *= 0.6
            else:
                link.base_bandwidth *= 1.2
                link.base_loss *= 0.7

    return meo_nodes, leo_nodes, links, links_dict, leo_to_meo, leo_neighbors

def get_link_metrics(node1, node2, links_dict):
    link = links_dict.get((node1, node2)) or links_dict.get((node2, node1))
    return (link.delay, link.bandwidth, link.loss, link) if link else (1000.0, 0.0, 100.0, None)

def calculate_shannon_capacity(bandwidth, snr):
    """使用香农定理计算理论最大带宽
    C = B * log2(1 + SNR)
    C: 信道容量 (bps)
    B: 带宽 (Hz)
    SNR: 信噪比 (线性值，非dB)
    """
    # 将dB转换为线性值
    snr_linear = 10 ** (snr / 10)
    # 计算信道容量 (Mbps)
    capacity = (bandwidth * 1e6 * np.log2(1 + snr_linear)) / 1e6
    return capacity

def calculate_link_metrics(node1, node2, links_dict, current_time=0):
    """计算链路性能指标
    
    参数:
        node1, node2: 链路两端节点
        links_dict: 链路字典
        current_time: 当前时间戳
    """
    link = links_dict.get((node1, node2)) or links_dict.get((node2, node1))
    if not link:
        return None

    if current_time - link.last_update_time >= UPDATE_INTERVAL:
        # 基础带宽计算
        utilization = min(1.0, link.traffic / QUEUE_CAPACITY if QUEUE_CAPACITY > 0 else 1.0)
        
        # 添加轨道动态因素
        orbital_factor = 1.0 + 0.2 * np.sin(2 * np.pi * current_time / (24 * 3600 * 1000))  # 24小时周期
        
        # 大气衰减影响
        elevation_angle = random.uniform(20, 90)  # 假设仰角在20-90度之间
        atmospheric_loss = 0.5 / np.sin(np.radians(elevation_angle))
        
        # 多普勒效应影响
        doppler_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_time / (90 * 60 * 1000))  # 90分钟轨道周期
        
        # 计算实际SNR
        base_snr = SNR_MAX * link.weather_factor * (1 - atmospheric_loss/10)
        interference_factor = 1 - (utilization * 0.3)
        effective_snr = base_snr * interference_factor * doppler_factor * orbital_factor
        effective_snr = max(SNR_MIN, min(SNR_MAX, effective_snr))
        
        # 使用香农定理计算理论带宽上限
        theoretical_bandwidth = calculate_shannon_capacity(BANDWIDTH, effective_snr)
        
        # 计算实际可用带宽
        available_bandwidth = theoretical_bandwidth * (1 - utilization)
        link.current_bandwidth = max(1.0, min(link.base_bandwidth, available_bandwidth))
        
        # 1. 基础传播延迟
        base_propagation_delay = link.base_delay
        
        # 2. 改进的队列延迟计算
        queue_size_bits = len(link.packets['in_queue']) * PACKET_SIZE * 8 * 1024  # 转换为bits
        actual_bandwidth = link.current_bandwidth * 1e6  # 转换为bps
        queue_delay = (queue_size_bits / actual_bandwidth) * 1000  # 转换为毫秒
        
        # 3. 处理延迟 (设备处理数据包的时间)
        processing_delay = 0.1 * len(link.packets['in_queue'])  # 每个包0.1ms的处理时间
        
        # 4. 拥塞对传播延迟的影响
        utilization = min(1.0, link.traffic / QUEUE_CAPACITY if QUEUE_CAPACITY > 0 else 1.0)
        congestion_factor = 1 + (utilization ** 2) * 0.5  # 二次方使高负载时影响更显著
        propagation_delay = base_propagation_delay * congestion_factor
        
        # 5. 设备负载对处理延迟的影响
        device_load = len(link.packets['in_queue']) / link.max_packets
        processing_delay *= (1 + device_load)
        
        # 6. 计算天气影响导致的延迟
        weather_delay = base_propagation_delay * (1 - link.weather_factor) * 0.2  # 天气因素导致的额外延迟
        
        # 7. 计算总延迟
        total_delay = (propagation_delay +  # 传播延迟
                      queue_delay +         # 队列延迟
                      processing_delay +    # 处理延迟
                      weather_delay)        # 天气影响
        
        # 8. 添加随机抖动
        jitter = np.random.normal(0, min(5, total_delay * 0.05))  # 抖动不超过总延迟的5%
        total_delay = max(base_propagation_delay, total_delay + jitter)
        
        # 计算丢包率
        base_loss = link.base_loss
        congestion_loss = (utilization ** 2) * 0.1
        snr_loss = max(0, (SNR_MAX - effective_snr) / SNR_MAX) * 0.05
        weather_loss = (1 - link.weather_factor) * 0.03
        queue_overflow_loss = max(0, (link.traffic - QUEUE_CAPACITY) / QUEUE_CAPACITY) * 0.1
        total_loss = min(0.99, base_loss + congestion_loss + snr_loss + weather_loss + queue_overflow_loss)
        
        # 更新链路状态
        link.current_bandwidth = link.current_bandwidth
        link.current_delay = total_delay
        link.current_loss = total_loss
        link.weather_factor = link.weather_factor
        link.last_update_time = current_time

    return {
        'delay': link.current_delay,
        'bandwidth': link.current_bandwidth,
        'loss': link.current_loss * 100,  # 转换为百分比
        'last_update': link.last_update_time
    }

def generate_traffic_poisson(mean_rate, time_interval=1.0, data_generation_rate=DATA_GENERATION_RATE):
    """生成泊松分布的流量,不限制生成数量
    
    参数:
        mean_rate: 平均速率 (Gbps)
        time_interval: 时间间隔 (秒)
        data_generation_rate: 数据生成率 (Gbps)
    """
    # 1. 将Gbps转换为每秒包数
    bits_per_second = data_generation_rate * 1e9
    packet_bits = PACKET_SIZE * 1024 * 8  # 每个包的比特数
    packets_per_second = bits_per_second / packet_bits
    
    # 2. 计算时间间隔内的期望包数
    lambda_packets = packets_per_second * time_interval
    
    # 3. 使用泊松分布生成实际数据包数
    # 不再限制最小值为1,让它自然反映数据生成率
    generated_packets = np.random.poisson(lambda_packets)
    
    # 4. 添加随机波动 (±10%)
    variation = random.uniform(0.9, 1.1)
    
    return int(generated_packets * variation)

def calculate_network_load(links_dict):
    """
    计算整体网络负载
    """
    total_load = 0
    total_links = 0
    
    for link in links_dict.values():
        if link.base_bandwidth > 0:
            load = link.traffic / link.base_bandwidth
            total_load += load
            total_links += 1
    
    return total_load / total_links if total_links > 0 else 1.0


def find_k_shortest_paths(source, destination, k, graph):
    """
    使用Yen算法找到k条最短路径
    """
    def dijkstra(start, end, graph, excluded_nodes=None):
        if excluded_nodes is None:
            excluded_nodes = set()

        distances = {node: float('infinity') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        unvisited = set(node for node in graph if node not in excluded_nodes)

        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if current == end:
                break

            unvisited.remove(current)

            for neighbor in graph[current]:
                if neighbor in excluded_nodes:
                    continue

                distance = distances[current] + 1  # 假设所有边的权重为1
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current

        if distances[end] == float('infinity'):
            return None, float('infinity')

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1], distances[end]

    def get_path_cost(path):
        return len(path) - 1  # 路径成本即为跳数

    # 存储找到的路径
    paths = []

    # 找第一条最短路径
    first_path, cost = dijkstra(source, destination, graph)
    if first_path is None:
        return paths
    paths.append((first_path, cost))

    # 候选路径
    candidates = []

    # 找剩余的k-1条路径
    for i in range(k - 1):
        prev_path = paths[-1][0]

        # 对前一条路径的每个节点
        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j]

            # 临时移除已有路径中的边
            excluded_nodes = set()
            for path, _ in paths:
                if len(path) > j and path[:j] == root_path:
                    if len(path) > j + 1:
                        excluded_nodes.add(path[j])

            # 对于root path中的节点，只保留第一个节点
            if j > 0:
                excluded_nodes.update(root_path[1:])

            # 找到从spur node到终点的路径
            spur_path, spur_cost = dijkstra(spur_node, destination, graph, excluded_nodes)

            if spur_path is not None:
                # 完整路径 = root路径 + spur路径
                total_path = root_path + spur_path
                total_cost = get_path_cost(total_path)
                candidates.append((total_path, total_cost))

        if not candidates:
            break

        # 选择最短的候选路径
        candidates.sort(key=lambda x: (x[1], x[0]))  # 按成本和字典序排序

        # 确保路径唯一
        while candidates and any(p[0] == candidates[0][0] for p in paths):
            candidates.pop(0)

        if candidates:
            paths.append(candidates.pop(0))

    return [path for path, _ in paths]

def plot_metrics(metrics, save_dir='training_plots'):
    """绘制训练指标图表"""
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取当前时间作为文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
    
    # 绘制各项指标
    episodes = range(1, len(metrics['delays']) + 1)
    
    # 延迟
    ax1.plot(episodes, metrics['delays'])
    ax1.set_title('平均延迟 (ms)')
    ax1.set_xlabel('训练轮次')
    ax1.grid(True)
    
    # 带宽
    ax2.plot(episodes, metrics['bandwidths'])
    ax2.set_title('平均带宽 (Mbps)')
    ax2.set_xlabel('训练轮次')
    ax2.grid(True)
    
    # 丢包率
    ax3.plot(episodes, metrics['losses'])
    ax3.set_title('丢包率')
    ax3.set_xlabel('训练轮次')
    ax3.grid(True)
    
    # 路径长度
    ax4.plot(episodes, metrics['path_lengths'])
    ax4.set_title('路径长度')
    ax4.set_xlabel('训练轮次')
    ax4.grid(True)
    
    # 传输成功率
    ax5.plot(episodes, metrics['transmission_success_rates'])
    ax5.set_title('传输成功率 (%)')
    ax5.set_xlabel('训练轮次')
    ax5.grid(True)
    
    # 奖励
    ax6.plot(episodes, metrics['rewards'])
    ax6.set_title('奖励值')
    ax6.set_xlabel('训练轮次')
    ax6.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{save_dir}/training_metrics_{timestamp}.png')
    plt.close()

def main():
    meo_nodes, leo_nodes, links, links_dict, leo_to_meo, leo_neighbors = setup_network()
    
    # 初始化可用动作缓存
    available_actions_cache = {}
    for leo_name in leo_nodes:
        available_actions = []
        for i, potential_next_leo in enumerate(leo_nodes):
            if (leo_name, potential_next_leo) in links_dict or (potential_next_leo, leo_name) in links_dict:
                available_actions.append(i)
        available_actions_cache[leo_name] = available_actions
    
    # 初始化MEO控制器
    meo_controllers = {}
    for meo_name in meo_nodes:
        managed_leos = {leo: leo_nodes[leo] for leo in leo_nodes if leo_to_meo[leo] == meo_name}
        meo_controllers[meo_name] = MEOController(meo_name, managed_leos)
    
    # 构建LEO网络图
    leo_graph = {}
    for leo_name in leo_nodes:
        leo_graph[leo_name] = set()
        for neighbor in leo_neighbors[leo_name]:
            leo_graph[leo_name].add(neighbor)
    
    # 修改：使用固定时间步长
    TIME_STEP = 20  # 固定为20ms
    simulation_time = 0  # 模拟时间（毫秒）
    
    # 添加时间更新间隔
    NETWORK_UPDATE_INTERVAL = 500  # 网络状态更新间隔保持500ms
    last_network_update = 0
    
    # 添加全局包ID计数器
    global_packet_id = 0
    
    # 添加数据生成率参数
    current_generation_rate = DATA_GENERATION_RATE  # 默认生成率（Gbps）
    
    state_size = 17
    action_size = len(leo_nodes)
    agent = DQNAgent(state_size, action_size, list(leo_nodes.keys()), leo_to_meo)
    leo_names = list(leo_nodes.keys())
    episodes = 2000
    batch_size = 32
    
    # 修改性能指标统计
    all_metrics = {
        'delays': [],
        'bandwidths': [],
        'losses': [],
        'path_lengths': [],
        'transmission_success_rates': [],
        'rewards': []
    }
    
    for episode in range(episodes):
        # 每个episode开始时重置所有链路的统计数据
        for link in links_dict.values():
            link.packets = {
                'in_queue': set(),      # 当前在队列中的包
                'processed': set(),      # 已处理的包
                'dropped': set(),        # 因队列满被丢弃的包
                'lost': set()           # 传输过程中丢失的包
            }
            link.packet_timestamps = {}  # 记录包的到达时间
            link.traffic = 0
        
        # 重置路径统计数据
        path_stats = {
            'sent': set(),          # 发送的所有包
            'dropped': set(),       # 因队列满被丢弃的包
            'lost': set(),          # 传输过程中丢失的包
            'received': set()       # 成功接收的包
        }
        
        source = random.choice(leo_names)
        destination = random.choice([x for x in leo_names if x != source])
        
        current_leo = source
        path = [current_leo]
        episode_reward = 0
        success = False
        steps = 0
        
        # 记录路径的性能指标
        path_delay = 0
        path_bandwidth = float('inf')
        
        # 初始化当前状态
        current_state = agent.get_state(current_leo, destination, links_dict, leo_nodes)
        
        while steps < MAX_PATH_LENGTH:
            simulation_time += TIME_STEP
            
            # 更新网络状态
            if simulation_time - last_network_update >= NETWORK_UPDATE_INTERVAL:
                for link in links_dict.values():
                    _ = calculate_link_metrics(link.node1.name, link.node2.name, links_dict, simulation_time)
                last_network_update = simulation_time
            
            steps += 1
            available_actions = available_actions_cache[current_leo]
            if not available_actions:
                break
                
            action = agent.choose_action(
                current_state, 
                available_actions, 
                current_leo, 
                destination, 
                meo_controllers, 
                links_dict,
                leo_graph,
                path
            )
            
            if action is None:
                break
                
            next_leo = leo_names[action]
            metrics = calculate_link_metrics(current_leo, next_leo, links_dict, simulation_time)
            
            if metrics:
                path_delay += metrics['delay']
                path_bandwidth = min(path_bandwidth, metrics['bandwidth'])
                
                if next_leo != destination:
                    link = links_dict.get((current_leo, next_leo)) or links_dict.get((next_leo, current_leo))
                    if link:
                        # 生成新的数据包
                        packets_to_send = generate_traffic_poisson(current_generation_rate, TIME_STEP / 1000.0)
                        new_packets = set(range(global_packet_id, global_packet_id + packets_to_send))
                        path_stats['sent'].update(new_packets)
                        
                        # 尝试添加到队列
                        accepted, dropped, global_packet_id = link.add_packets(
                            packets_to_send, global_packet_id, simulation_time
                        )
                        path_stats['dropped'].update(dropped)
                        
                        # 处理队列中的数据包
                        processed = link.process_queue(simulation_time)
                        
                        # 计算传输丢失
                        lost = link.calculate_packet_loss(processed, metrics['loss'])
                        path_stats['lost'].update(lost)
                        
                        # 更新成功接收的包
                        path_stats['received'].update(processed - lost)
                
                # 计算奖励
                if next_leo == destination:
                    success = True
                    reward = 20.0 + max(0, (MAX_PATH_LENGTH - len(path)) * 0.5)
                else:
                    reward = -0.05
                    if leo_to_meo[next_leo] == leo_to_meo[destination]:
                        reward += 1.0
                    if next_leo in path:
                        reward -= 0.5
                
                next_state = agent.get_state(next_leo, destination, links_dict, leo_nodes)
                agent.memorize(current_state, action, reward, next_state, next_leo == destination)
                
                current_state = next_state
                current_leo = next_leo
                path.append(current_leo)
                episode_reward += reward
                
                if next_leo == destination:
                    break
        
        # 打印每个episode的统计信息
        print(f"Episode {episode + 1}/{episodes}")
        print(f"路径: {' -> '.join(path)}")
        print(f"发送包数: {len(path_stats['sent'])}")
        print(f"接收包数: {len(path_stats['received'])}")
        print(f"传输丢失包数: {len(path_stats['lost'])}")
        print(f"队列丢弃包数: {len(path_stats['dropped'])}")
        
        total_sent = len(path_stats['sent'])
        if total_sent > 0:
            # 计算各种包的数量
            dropped_packets = len(path_stats['dropped'])  # 队列满丢弃的包
            lost_packets = len(path_stats['lost'])       # 传输过程丢失的包
            received_packets = len(path_stats['received']) # 成功接收的包
            
            # 计算在途包数（仍在队列中的包）
            in_transit_packets = 0
            for link in links_dict.values():
                in_transit_packets += len(link.packets['in_queue'])
            
            # 计算各种比率
            drop_rate = (dropped_packets / total_sent) * 100
            loss_rate = (lost_packets / total_sent) * 100
            success_rate = (received_packets / total_sent) * 100
            in_transit_rate = (in_transit_packets / total_sent) * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"路径: {' -> '.join(path)}")
            print(f"带宽：{path_bandwidth:.2f} Mbps")
            print(f"延迟：{path_delay:.2f} ms")
            print(f"发送包数: {total_sent}")
            print(f"接收包数: {received_packets}")
            print(f"传输丢失包数: {lost_packets}")
            print(f"队列丢弃包数: {dropped_packets}")
            print(f"在途包数: {in_transit_packets}")
            print(f"队列丢弃率: {drop_rate:.2f}%")
            print(f"传输丢失率: {loss_rate:.2f}%")
            print(f"总丢包率: {(drop_rate + loss_rate):.2f}%")
            print(f"传输成功率: {success_rate:.2f}%")
            print(f"在途率: {in_transit_rate:.2f}%")
            print(f"统计完整性检查: {(drop_rate + loss_rate + success_rate + in_transit_rate):.2f}%")
            
            # 确保所有包都被追踪到
            total_accounted = dropped_packets + lost_packets + received_packets + in_transit_packets
            if total_accounted != total_sent:
                print(f"警告：包计数不匹配！总计：{total_accounted}，发送：{total_sent}")
        
        print(f"路径长度: {len(path)}")
        print(f"奖励: {episode_reward:.2f}")
        print(f"探索率: {agent.epsilon:.3f}")
        print("-" * 50)
        
        # 更新指标记录
        all_metrics['delays'].append(path_delay)
        all_metrics['bandwidths'].append(path_bandwidth)
        all_metrics['losses'].append(loss_rate)
        all_metrics['path_lengths'].append(len(path))
        all_metrics['transmission_success_rates'].append(success_rate)
        all_metrics['rewards'].append(episode_reward)
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        if episode % 10 == 0:
            agent.update_target_network()
    
    # 训练结束后保存图表
    plot_metrics(all_metrics)

if __name__ == '__main__':
    main()