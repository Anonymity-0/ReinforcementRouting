from typing import Dict, List, Optional, Any, Tuple
import heapq
import time
import logging
import gym
import numpy as np
from simulation.event import Event, EventType
from .tle_constellation import TLEConstellation
from .traffic_generator import TrafficGenerator
from algorithms.routing_interface import RoutingAlgorithm
from algorithms.dijkstra_algorithm import DijkstraAlgorithm
import yaml
import random
import math
from .packet import Packet

class NetworkSimulator(gym.Env):
    """网络仿真器,同时支持事件驱动和强化学习训练"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化网络仿真器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        
        print("开始初始化NetworkSimulator...")
        
        self.config = config
        self.current_time = 0.0
        self.next_metrics_time = 0.0
        
        # 从配置中获取基本参数
        self.step_interval = config['simulation']['common']['step_interval']
        self.metrics_interval = config['simulation']['common']['metrics_collection_interval']
        self.max_steps = config['simulation']['common']['max_steps']
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 初始化拓扑
        self.topology = TLEConstellation(config)
        self.total_satellites = self.topology.total_satellites
        
        # 初始化流量生成器
        self.traffic_generator = TrafficGenerator(config)
        
        # 初始化事件队列
        self.event_queue = []
        
        # 初始化时间
        self.current_step = 0
        
        # 初始化观察和动作空间
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # 初始化路由算法
        self.routing_algorithm = None
        
        # 初始化指标收集器
        self.metrics = {
            'delay': [],
            'throughput': [],
            'packet_loss': []
        }
        
        print("NetworkSimulator初始化完成")
    
    def step(self, action=None):
        """
        执行一步仿真
        
        Args:
            action: 路由动作(可选)
            
        Returns:
            Tuple: (下一个状态, 奖励, 是否结束, 信息字典)
        """
        # 更新当前时间
        self.current_time += self.step_interval
        self.current_step += 1
        
        # 处理数据包生成
        # 根据泊松分布计算这个时间步应该生成的数据包数量
        lambda_t = self.config['simulation']['traffic']['poisson_lambda'] * self.step_interval
        num_packets = np.random.poisson(lambda_t)
        
        # 生成数据包
        for _ in range(num_packets):
            packet = self.traffic_generator.generate_packet(self.current_time)
            # 更新源节点的数据包计数
            self.topology.satellite_objects[packet.source].total_packets += 1
            # 使用M/M/1/N队列模型处理数据包到达
            self._schedule_event(EventType.PACKET_ARRIVAL, self.current_time, {
                'packet': packet,
                'current_node': packet.source
            })
        
        # 处理链路更新
        if self.current_time >= self.topology.next_link_update:
            self.topology.update(self.current_time)
            self.topology.next_link_update = self.current_time + self.config['simulation']['link']['update_interval']
        
        # 处理所有当前时间的事件
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            _, _, event = heapq.heappop(self.event_queue)
            self._process_event(event)
        
        # 收集指标
        metrics = self._collect_metrics()
        
        # 计算奖励
        reward = self._calculate_reward(metrics, action)
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        return self.get_state(), reward, done, {'metrics': metrics}
        
    def _process_event(self, event: Event):
        """
        处理事件
        
        Args:
            event: 事件对象
        """
        if event.type == EventType.PACKET_GENERATION:
            packet = self.traffic_generator.generate_packet(self.current_time)
            self._schedule_event(EventType.PACKET_ARRIVAL, self.current_time, {
                'packet': packet,
                'current_node': packet.source
            })
        elif event.type == EventType.PACKET_ARRIVAL:
            self._handle_packet_arrival(event.data)
        elif event.type == EventType.PACKET_SERVICE:
            self._handle_packet_service(event.data)
        elif event.type == EventType.LINK_UPDATE:
            self.topology.update(self.current_time)
        elif event.type == EventType.TOPOLOGY_UPDATE:
            self.topology.update(self.current_time)
            
    def _schedule_event(self, event_type: EventType, time: float, data: Any = None):
        """
        调度新事件
        
        Args:
            event_type: 事件类型
            time: 事件发生时间
            data: 事件数据
        """
        event = Event(event_type, time, data)
        heapq.heappush(self.event_queue, (time, len(self.event_queue), event))
        
    def _create_observation_space(self) -> gym.Space:
        """创建观测空间"""
        num_satellites = self.total_satellites
        
        # 计算状态维度
        # 1. 当前节点特征 (3 + 3 + 1 + 1 = 8)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        #    - 轨道面 (1)
        # 2. 目标节点特征 (3 + 3 + 1 + 1 = 8)
        #    - 位置 (3)
        #    - 速度 (3)
        #    - 队列长度 (1)
        #    - 轨道面 (1)
        # 3. 链路状态 (total_satellites)
        # 4. 全局特征 (3)
        #    - 当前节点ID (1)
        #    - 目标节点ID (1)
        #    - 轨道面距离 (1)
        state_dim = 8 + 8 + num_satellites + 3
        
        # 创建Box空间，确保维度正确
        return gym.spaces.Box(
            low=np.array([-np.inf] * state_dim),
            high=np.array([np.inf] * state_dim),
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def _create_action_space(self) -> gym.Space:
        """创建动作空间"""
        num_satellites = self.total_satellites
        # 动作空间为当前节点的邻居卫星集合
        # 使用Discrete(num_satellites + 1)表示可能的下一跳选择
        # 其中num_satellites表示保持在当前节点(当没有可用邻居时)
        return gym.spaces.Discrete(num_satellites + 1)
    
    def _get_valid_actions(self, current_node: int) -> List[int]:
        """
        获取当前节点的有效动作(可以建立连接的邻居节点)
        
        Args:
            current_node: 当前节点编号
            
        Returns:
            valid_neighbors: 可以建立连接的邻居节点列表
        """
        valid_neighbors = self.topology.get_valid_neighbors(current_node)
        
        if not valid_neighbors:
            return [-1]
            
        return valid_neighbors
    
    def reset(self):
        """
        重置环境状态
        
        Returns:
            Dict: 初始状态
        """
        # 重置当前步数
        self.current_step = 0
        
        # 重置拓扑
        self.topology.reset()
        
        # 重置统计指标
        self.metrics = {
            'delay': [],
            'throughput': [],
            'packet_loss': [],
            'packets_generated': 0,
            'packets_delivered': 0,
            'packets_dropped': 0
        }
        
        # 随机选择源节点和目标节点
        num_satellites = self.total_satellites
        self.source = np.random.randint(0, num_satellites)
        self.destination = np.random.randint(0, num_satellites)
        while self.destination == self.source:
            self.destination = np.random.randint(0, num_satellites)
        self.current_node = self.source
        
        # 获取初始状态
        state = self.get_state()
        
        return state
        
    def get_state(self) -> Dict:
        """
        获取当前状态
            
        Returns:
            Dict: 状态字典，包含完整的状态信息
        """
        # 获取拓扑信息
        positions = self.topology.positions  # [num_satellites, 3]
        velocities = self.topology.velocities  # [num_satellites, 3]
        
        # 获取网络信息
        num_satellites = self.total_satellites
        queue_lengths = np.zeros(num_satellites, dtype=np.float32)
        for i in range(num_satellites):
            queue_lengths[i] = len(self.topology.satellite_objects[i].queue)
            
        # 获取链路状态
        link_states = np.zeros(num_satellites, dtype=np.float32)
        for i in range(num_satellites):
            link_data = self.topology.get_link_data(i, self.current_node)
            link_states[i] = link_data['quality'] if link_data else 0.0
                    
        # 如果是第一步，随机选择源节点和目标节点
        if self.current_step == 0:
            self.source = np.random.randint(0, num_satellites)
            self.destination = np.random.randint(0, num_satellites)
            while self.destination == self.source:
                self.destination = np.random.randint(0, num_satellites)
            self.current_node = self.source
            
        # 构造状态字典
        state = {
            'topology': {
                'positions': positions,
                'velocities': velocities,
                'object': self.topology  # 添加topology对象作为子字段
            },
            'network': {
                'queue_lengths': queue_lengths,
                'link_states': link_states
            },
            'packet': {
                'source': self.source,
                'destination': self.destination,
                'current_node': self.current_node,
                'size': np.array([self.config['simulation']['traffic']['packet_size']], dtype=np.float32),
                'qos_class': 0  # 默认使用第一个QoS类别
            }
        }
        
        return state
        
    def _calculate_link_capacity(self, link):
        """使用Shannon公式计算链路容量"""
        # 获取链路参数
        frequency = self.config['topology']['link']['frequency']
        power = self.config['topology']['link']['transmit_power']
        noise_temp = self.config['topology']['link']['noise_temperature']
        bandwidth = self.config['topology']['link']['bandwidth']
        
        # 计算路径损耗
        path_loss = self._calculate_path_loss(link.distance, frequency)
        
        # 计算信噪比
        k = 1.38e-23  # 玻尔兹曼常数
        noise_power = k * noise_temp * bandwidth
        snr = (power / path_loss) / noise_power
        
        # 使用Shannon公式计算容量
        capacity = bandwidth * math.log2(1 + snr)
        
        return capacity
        
    def _calculate_path_loss(self, distance, frequency):
        """计算自由空间路径损耗"""
        c = 3e8  # 光速
        wavelength = c / frequency
        
        # 自由空间路径损耗公式
        path_loss = (4 * math.pi * distance * 1000 / wavelength) ** 2
        
        return path_loss
        
    def _calculate_link_delay(self, link):
        """
        计算链路延迟
        
        Args:
            link: 链路数据
            
        Returns:
            float: 总延迟(秒)
        """
        # 如果链路不可用
        if link['quality'] <= 0:
            return float('inf')
            
        # 计算传播延迟
        propagation_delay = link['distance'] / 3e5  # 光速传播
        
        # 获取链路容量
        capacity = link['capacity']  # bits per second
        
        # 计算传输延迟
        packet_size = self.config['simulation']['traffic']['packet_size'] * 8  # bits
        transmission_delay = packet_size / capacity
        
        # 使用M/M/1/N队列模型计算队列延迟
        # 获取到达率(packets/second)
        arrival_rate = self.config['simulation']['traffic']['poisson_lambda']
        # 获取服务率(packets/second)
        service_rate = capacity / packet_size
        # 获取最大队列长度
        max_queue_length = self.config['simulation']['network']['max_queue_length']
        # 获取当前队列长度
        current_queue_length = len(self.topology.satellite_objects[link['dst']].queue)
        
        # 计算利用率
        rho = arrival_rate / service_rate
        
        # 计算M/M/1/N队列的平均队列长度
        if rho != 1:
            L = (rho * (1 - rho ** (max_queue_length + 1))) / (1 - rho ** (max_queue_length + 2))
        else:
            L = max_queue_length / 2
            
        # 使用Little's Law计算队列延迟
        if arrival_rate > 0:
            queuing_delay = L / arrival_rate
        else:
            queuing_delay = 0
            
        # 返回总延迟
        total_delay = propagation_delay + transmission_delay + queuing_delay
        return max(0.0, total_delay)  # 确保延迟非负
        
    def _handle_packet_arrival(self, data: Dict):
        """
        处理数据包到达事件
        
        Args:
            data: 事件数据
        """
        # 获取数据包和当前节点
        packet = data['packet']
        current_node = data['current_node']
        
        # 如果是第一次进入网络，设置创建时间
        if packet.create_time is None:
            packet.create_time = self.current_time
        
        # 如果已到达目标节点
        if current_node == packet.destination:
            # 更新统计信息
            self.topology.satellite_objects[current_node].delivered_packets += 1
            self.topology.satellite_objects[current_node].total_bytes_delivered += packet.size
            packet.is_delivered = True
            packet.deliver_time = self.current_time
            
            # 记录延迟
            delay = packet.get_delay()
            if delay > 0:
                self.topology.satellite_objects[current_node].packet_delays.append(delay)
            
            return
            
        # 使用M/M/1/N队列模型处理数据包入队
        satellite = self.topology.satellite_objects[current_node]
        if satellite.enqueue_packet(packet):
            # 如果入队成功，调度服务完成事件
            service_time = np.random.exponential(1.0 / satellite.service_rate)
            self._schedule_event(EventType.PACKET_SERVICE, self.current_time + service_time, {
                'packet': packet,
                'current_node': current_node
            })
        else:
            # 如果入队失败，更新丢包统计
            satellite.dropped_packets += 1
            
    def _handle_packet_service(self, data: Dict):
        """
        处理数据包服务完成事件
        
        Args:
            data: 事件数据
        """
        packet = data['packet']
        current_node = data['current_node']
        satellite = self.topology.satellite_objects[current_node]
        
        # 获取路由决策
        state = self.get_state()
        state['packet']['current_node'] = current_node
        state['packet']['destination'] = packet.destination
        
        next_hop = self.routing_algorithm.get_next_hop(current_node, packet.destination, state)
        
        # 如果已到达目的地
        if next_hop == packet.destination:
            packet.is_delivered = True
            packet.deliver_time = self.current_time
            satellite.delivered_packets += 1
            satellite.total_bytes_delivered += packet.size
            
            # 计算延迟并添加到统计信息
            delay = packet.get_delay()
            if delay > 0:
                satellite.packet_delays.append(delay)
                
            # 从队列中移除数据包
            satellite.dequeue_packet()
            return
            
        # 如果找不到路径,记录为路由丢失
        if next_hop == -1:
            satellite.lost_packets += 1
            packet.drop_reason = 'no_valid_path'
            satellite.dequeue_packet()
            return
            
        # 检查链路质量
        link_data = self.topology.get_link_data(current_node, next_hop)
        next_satellite = self.topology.satellite_objects[next_hop]
        
        # 检查链路质量
        if link_data['quality'] <= 0:
            satellite.lost_packets += 1
            packet.drop_reason = 'link_failure'
            satellite.dequeue_packet()
            return
            
        # 检查下一跳缓冲区
        if len(next_satellite.queue) >= next_satellite.max_queue_length:
            satellite.dropped_packets += 1
            packet.drop_reason = 'buffer_overflow'
            satellite.dequeue_packet()
            return
            
        # 正常转发到下一跳
        satellite.dequeue_packet()
        # 更新路由信息
        packet.update_route(next_hop, self.current_time)
        # 调度到达下一跳的事件
        propagation_delay = link_data['distance'] / 3e5
        self._schedule_event(EventType.PACKET_ARRIVAL, self.current_time + propagation_delay, {
            'packet': packet,
            'current_node': next_hop
        })
        
    def _forward_packets(self):
        """转发数据包"""
        # 获取所有卫星
        num_satellites = self.total_satellites
        
        # 对每个卫星进行数据包转发
        for i in range(num_satellites):
            satellite = self.topology.satellite_objects[i]
            
            # 获取待转发的数据包
            packets = list(satellite.queue)
            
            for packet in packets:
                # 使用路由算法获取下一跳
                state = self.get_state()
                state['packet']['current_node'] = i
                state['packet']['destination'] = packet.destination
                
                next_hop = self.routing_algorithm.get_next_hop(i, packet.destination, state)
                
                # 如果已到达目的地
                if next_hop == 0:
                    # 标记为已送达并更新统计信息
                    packet.is_delivered = True
                    packet.deliver_time = self.current_time
                    satellite.delivered_packets += 1
                    satellite.total_bytes_delivered += packet.size
                    
                    # 计算延迟并添加到统计信息
                    delay = packet.get_delay()
                    if delay > 0:  # 只记录有效的延迟值
                        satellite.packet_delays.append(delay)
                    
                    # 从队列中移除
                    if packet in satellite.queue:
                        satellite.queue.remove(packet)
                    continue
                
                # 如果找不到路径,直接丢包
                if next_hop == -1:
                    satellite.lost_packets += 1
                    if packet in satellite.queue:
                        satellite.queue.remove(packet)
                    continue
                
                # 正常转发到下一跳
                # 检查链路容量和缓冲区
                link_data = self.topology.get_link_data(i, next_hop)
                next_satellite = self.topology.satellite_objects[next_hop]
                
                if (link_data['quality'] > 0 and 
                    len(next_satellite.queue) < self.config['simulation']['network']['max_queue_length']):
                    
                    # 转发数据包
                    packet.update_route(next_hop, self.current_time)
                    next_satellite.enqueue_packet(packet)
                    
                    # 从当前节点的队列中移除数据包
                    if packet in satellite.queue:
                        satellite.queue.remove(packet)
                else:
                    # 链路不可用或缓冲区已满,直接丢包
                    satellite.lost_packets += 1
                    if packet in satellite.queue:
                        satellite.queue.remove(packet)
        
    def _get_next_hop(self, current_node: int, packet: 'Packet') -> int:
        """
        获取数据包的下一跳节点
        
        Args:
            current_node: 当前节点
            packet: 数据包
            
        Returns:
            int: 下一跳节点ID，如果没有可用的下一跳则返回None
        """
        # 获取当前状态
        state = self.get_state()
        state['packet']['current_node'] = current_node
        state['packet']['destination'] = packet.destination
        
        if self.routing_algorithm:
            return self.routing_algorithm.get_next_hop(
                current_node,
                packet.destination,
                state
            )
        else:
            # 简单的贪心路由：选择到目标节点距离最近的邻居
            num_satellites = self.total_satellites
            best_next_hop = None
            min_distance = float('inf')
            
            # 遍历所有可能的下一跳
            for next_hop in range(num_satellites):
                if next_hop != current_node:
                    # 检查链路是否可用
                    link_data = self.topology.get_link_data(current_node, next_hop)
                    if link_data['quality'] > 0:
                        # 计算到目标的距离
                        next_pos = self.topology.positions[next_hop]
                        dest_pos = self.topology.positions[packet.destination]
                        distance = np.linalg.norm(next_pos - dest_pos)
                        
                        # 更新最佳下一跳
                        if distance < min_distance:
                            min_distance = distance
                            best_next_hop = next_hop
            
            return best_next_hop
        
    def _collect_metrics(self):
        """收集性能指标"""
        current_time = self.current_step * self.step_interval
        
        # 初始化统计变量
        total_packets = 0
        delivered_packets = 0
        lost_packets = 0
        dropped_packets = 0
        total_bytes_delivered = 0
        total_bytes_received = 0
        delays = []
        queue_lengths = []
        
        # 丢包原因统计
        drop_reasons = {
            'no_valid_path': 0,
            'link_failure': 0,
            'buffer_overflow': 0
        }
        
        # 统计所有节点的指标
        for i in range(self.total_satellites):
            satellite = self.topology.satellite_objects[i]
            
            # 基本统计
            total_packets += satellite.total_packets
            delivered_packets += satellite.delivered_packets
            lost_packets += satellite.lost_packets
            dropped_packets += satellite.dropped_packets
            total_bytes_delivered += satellite.total_bytes_delivered
            total_bytes_received += satellite.total_bytes_received
            
            # 统计丢包原因
            for packet in satellite.dropped_packets_history:
                if hasattr(packet, 'drop_reason'):
                    drop_reasons[packet.drop_reason] = drop_reasons.get(packet.drop_reason, 0) + 1
            
            # 延迟统计
            if satellite.packet_delays:
                delays.extend(satellite.packet_delays)
            
            # 使用M/M/1/N队列理论计算队列长度
            queue_lengths.append(satellite.get_average_queue_length(current_time))
        
        # 计算当前指标
        if current_time > 0:
            network_throughput = sum(satellite.get_throughput(current_time) 
                                   for satellite in self.topology.satellite_objects)
            avg_satellite_throughput = network_throughput / self.total_satellites
        else:
            network_throughput = 0
            avg_satellite_throughput = 0
            
        # 计算当前延迟
        current_delay = np.mean(delays) if delays else 0
        min_delay = np.min(delays) if delays else 0
        max_delay = np.max(delays) if delays else 0
        
        # 计算当前交付率和丢包率
        total_lost = lost_packets + dropped_packets
        if total_packets > 0:
            delivery_rate = delivered_packets / total_packets
            packet_loss_rate = total_lost / total_packets
        else:
            delivery_rate = 0
            packet_loss_rate = 0
        
        # 计算当前队列长度
        avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0
        
        # 更新指标
        metrics = {
            'delay': current_delay,
            'min_delay': min_delay,
            'max_delay': max_delay,
            'throughput': network_throughput,
            'packet_loss': packet_loss_rate,
            'packets_generated': total_packets,
            'packets_delivered': delivered_packets,
            'packets_dropped': total_lost,
            'delivery_rate': delivery_rate,
            'avg_queue_length': avg_queue_length,
            'avg_satellite_throughput': avg_satellite_throughput,
            'drop_reasons': drop_reasons,
            'buffer_overflow': any(len(sat.queue) >= sat.max_queue_length for sat in self.topology.satellite_objects),
            'link_failure': False  # 将在每个包处理时单独判断
        }
        
        # 更新类成员变量中的指标
        self.metrics['delay'].append(current_delay)
        self.metrics['throughput'].append(network_throughput)
        self.metrics['packet_loss'].append(packet_loss_rate)
        
        
        return metrics

    def run_dijkstra(self, source: int, destination: int) -> List[int]:
        """
        运行Dijkstra算法计算最短路径
        
        Args:
            source: 源节点
            destination: 目标节点
            
        Returns:
            List[int]: 计算得到的路径
        """
        if not self.routing_algorithm:
            raise ValueError("未设置路由算法")
        
        # 获取当前状态
        state = self.get_state()
        
        # 使用Dijkstra算法计算下一跳
        path = []
        current_node = source
        
        while current_node != destination:
            next_hop = self.routing_algorithm.get_next_hop(current_node, destination, state)
            
            # 如果找不到路径
            if next_hop == -1:
                return []
            
            path.append(current_node)
            current_node = next_hop
        
        path.append(destination)
        return path

    def set_routing_algorithm(self, algorithm):
        """
        设置路由算法
        
        Args:
            algorithm: 路由算法实例
        """
        self.routing_algorithm = algorithm

    def _can_establish_link(self, src: int, dst: int) -> bool:
        """
        判断两颗卫星是否可以建立链路
        
        Args:
            src: 源节点ID
            dst: 目标节点ID
            
        Returns:
            bool: 如果可以建立链路返回True，否则返回False
        """
        # 使用TLEConstellation的_can_establish_link方法
        return self.topology._can_establish_link(src, dst)

    def _calculate_reward(self, metrics: Dict[str, Any], action: int) -> float:
        """计算奖励值
        
        奖励函数设计:
        R = w₁R_d + w₂R_t + w₃R_l + w₄R_b
        
        其中:
        - R_d: 延迟奖励项 = -min(d/d_max, 1)
        - R_t: 吞吐量奖励项 = t/t_max
        - R_l: 丢包率惩罚项 = -min(l/l_max, 1)
        - R_b: 缓冲区使用率惩罚项 = -min(b/b_max, 1)
        
        权重:
        w₁ = 0.3 (延迟权重)
        w₂ = 0.3 (吞吐量权重)
        w₃ = 0.2 (丢包率权重)
        w₄ = 0.2 (缓冲区权重)
        
        Args:
            metrics: 性能指标字典
            action: 选择的动作
            
        Returns:
            float: 奖励值 ∈ [-1, 1]
        """
        # 获取配置参数
        d_max = self.config['simulation']['traffic']['qos_classes'][0]['delay_threshold']  # 最大可接受延迟
        t_max = self.config['simulation']['traffic']['qos_classes'][2]['throughput_threshold']  # 目标吞吐量
        l_max = self.config['simulation']['traffic']['qos_classes'][1]['loss_threshold']  # 最大可接受丢包率
        b_max = self.config['simulation']['network']['max_queue_length']  # 最大队列长度
        
        # 计算归一化的指标值
        delay = metrics.get('delay', 0)
        throughput = metrics.get('throughput', 0)
        packet_loss = metrics.get('packet_loss', 0)
        buffer_usage = metrics['queue_lengths'][action] if 'queue_lengths' in metrics else 0
        
        # 计算各个奖励项
        R_d = -min(delay / d_max, 1.0) if delay > 0 else 0  # 延迟奖励
        R_t = min(throughput / t_max, 1.0) if throughput > 0 else 0  # 吞吐量奖励
        R_l = -min(packet_loss / l_max, 1.0) if packet_loss > 0 else 0  # 丢包率惩罚
        R_b = -min(buffer_usage / b_max, 1.0)  # 缓冲区使用惩罚
        
        # 权重设置
        w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
        
        # 计算总奖励
        reward = w1 * R_d + w2 * R_t + w3 * R_l + w4 * R_b
        
        return reward 