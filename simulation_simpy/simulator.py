import simpy
import gym
import numpy as np
from typing import Dict, Any, Tuple, List
from simulation.tle_constellation import TLEConstellation
from simulation.traffic_generator import TrafficGenerator
from simulation.packet import Packet
from algorithms.routing_interface import RoutingAlgorithm

class SimpyNetworkSimulator(gym.Env):
    """基于SimPy的网络仿真器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化仿真器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        print("开始初始化SimpyNetworkSimulator...")
        
        self.config = config
        
        # 创建SimPy环境
        self.env = simpy.Environment()
        
        # 初始化拓扑
        self.topology = TLEConstellation(config)
        self.total_satellites = self.topology.total_satellites
        
        # 初始化流量生成器
        self.traffic_generator = TrafficGenerator(config)
        
        # 初始化资源
        self.satellite_queues = {}  # 卫星队列资源
        self.link_capacities = {}   # 链路容量资源
        
        # 初始化路由算法
        self.routing_algorithm = None
        
        # 初始化观察和动作空间
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # 初始化指标
        self.metrics = {
            'delay': [],
            'throughput': [],
            'packet_loss': [],
            'packets_generated': 0,
            'packets_delivered': 0,
            'packets_dropped': 0
        }
        
        # 初始化资源
        self._initialize_resources()
        
        # 启动核心进程
        self.packet_gen_process = self.env.process(self._packet_generation_process())
        self.link_update_process = self.env.process(self._link_update_process())
        
        print("SimpyNetworkSimulator初始化完成")
        
    def _initialize_resources(self):
        """初始化SimPy资源"""
        # 为每个卫星创建队列资源
        for sat_id in range(self.total_satellites):
            self.satellite_queues[sat_id] = simpy.Resource(
                self.env, 
                capacity=self.config['simulation']['network']['max_queue_length']
            )
            
        # 为每个可能的链路创建容量资源
        bandwidth = float(self.config['simulation']['link']['bandwidth'])  # 转换为浮点数
        for i in range(self.total_satellites):
            for j in range(i+1, self.total_satellites):
                if self.topology._can_establish_link(i, j):
                    self.link_capacities[(i,j)] = simpy.Container(
                        self.env,
                        capacity=bandwidth,
                        init=bandwidth
                    )
                    
    def _packet_generation_process(self):
        """数据包生成进程"""
        while True:
            # 生成数据包的时间间隔服从泊松分布
            interval = 1.0 / self.config['simulation']['traffic']['poisson_lambda']
            yield self.env.timeout(interval)
            
            # 生成新数据包
            packet = self.traffic_generator.generate_packet(self.env.now)
            self.metrics['packets_generated'] += 1
            
            # 启动数据包传输进程
            self.env.process(self._packet_transmission_process(packet))
            
    def _packet_transmission_process(self, packet: Packet):
        """数据包传输进程"""
        current_node = packet.source
        
        while current_node != packet.destination:
            # 获取下一跳
            next_hop = self._get_next_hop(current_node, packet.destination)
            if next_hop == -1:
                # 无法找到路径，丢弃数据包
                self._record_packet_drop(packet, "no_valid_path")
                break
                
            # 请求队列资源
            with self.satellite_queues[current_node].request() as queue_req:
                yield queue_req
                
                # 计算传输延迟
                link_delay = self._calculate_link_delay(current_node, next_hop)
                yield self.env.timeout(link_delay)
                
                # 更新指标
                self._update_metrics(packet, current_node, next_hop)
                
                # 移动到下一跳
                current_node = next_hop
                
        if current_node == packet.destination:
            self._record_packet_delivery(packet)
            
    def _link_update_process(self):
        """链路更新进程"""
        while True:
            yield self.env.timeout(self.config['simulation']['link']['update_interval'])
            self.topology.update(self.env.now)
            
    def _get_next_hop(self, current_node: int, destination: int) -> int:
        """获取下一跳节点"""
        if self.routing_algorithm:
            return self.routing_algorithm.get_next_hop(current_node, destination, self.get_state())
        return -1
        
    def _calculate_link_delay(self, src: int, dst: int) -> float:
        """计算链路延迟"""
        link_data = self.topology.get_link_data(src, dst)
        return link_data['delay']
        
    def _update_metrics(self, packet: Packet, src: int, dst: int):
        """更新性能指标"""
        link_data = self.topology.get_link_data(src, dst)
        
        # 更新吞吐量
        self.metrics['throughput'].append(link_data['capacity'])
        
        # 更新延迟
        if packet.create_time is not None:
            delay = self.env.now - packet.create_time
            self.metrics['delay'].append(delay)
            
    def _record_packet_drop(self, packet: Packet, reason: str):
        """记录数据包丢弃"""
        self.metrics['packets_dropped'] += 1
        self.metrics['packet_loss'].append(1.0)
        
    def _record_packet_delivery(self, packet: Packet):
        """记录数据包成功传输"""
        self.metrics['packets_delivered'] += 1
        self.metrics['packet_loss'].append(0.0)
        
    def step(self, action=None):
        """执行一步仿真"""
        # 运行SimPy环境一个时间步
        self.env.run(until=self.env.now + self.config['simulation']['common']['step_interval'])
        
        # 获取状态和指标
        state = self.get_state()
        metrics = self.get_metrics()
        
        # 计算奖励
        reward = self._calculate_reward(metrics, action)
        
        # 检查是否结束
        done = self.env.now >= self.config['simulation']['common']['max_steps']
        
        return state, reward, done, {'metrics': metrics}
        
    def reset(self):
        """重置环境"""
        # 重新创建SimPy环境
        self.env = simpy.Environment()
        
        # 重置拓扑
        self.topology.reset()
        
        # 重置指标
        self.metrics = {
            'delay': [],
            'throughput': [],
            'packet_loss': [],
            'packets_generated': 0,
            'packets_delivered': 0,
            'packets_dropped': 0
        }
        
        # 重新初始化资源
        self._initialize_resources()
        
        # 重新启动核心进程
        self.packet_gen_process = self.env.process(self._packet_generation_process())
        self.link_update_process = self.env.process(self._link_update_process())
        
        return self.get_state()
        
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
        
        return gym.spaces.Box(
            low=np.array([-np.inf] * state_dim),
            high=np.array([np.inf] * state_dim),
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def _create_action_space(self) -> gym.Space:
        """创建动作空间"""
        return gym.spaces.Discrete(self.total_satellites + 1)
        
    def get_state(self) -> Dict:
        """获取当前状态"""
        # 获取当前正在处理的数据包信息
        current_packet = None
        for packet in self.env.active_process.packets if hasattr(self.env.active_process, 'packets') else []:
            if not packet.delivered and not packet.dropped:
                current_packet = packet
                break
                
        # 构建状态字典
        state = {
            'topology': {
                'object': self.topology,
                'positions': self.topology.positions,
                'velocities': self.topology.velocities
            },
            'network': {
                'queue_lengths': [len(q.queue) for q in self.satellite_queues.values()],
                'link_states': [[1 if (i,j) in self.link_capacities else 0 
                               for j in range(self.total_satellites)]
                              for i in range(self.total_satellites)]
            },
            'packet': {
                'current_node': current_packet.current_node if current_packet else 0,
                'destination': current_packet.destination if current_packet else 0,
                'creation_time': current_packet.create_time if current_packet else self.env.now,
                'size': current_packet.size if current_packet else 0
            }
        }
        return state
        
    def get_metrics(self) -> Dict:
        """获取当前性能指标"""
        return {
            'delay': np.mean(self.metrics['delay']) if self.metrics['delay'] else 0.0,
            'throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0.0,
            'packet_loss': np.mean(self.metrics['packet_loss']) if self.metrics['packet_loss'] else 0.0,
            'packets_generated': self.metrics['packets_generated'],
            'packets_delivered': self.metrics['packets_delivered'],
            'packets_dropped': self.metrics['packets_dropped']
        }
        
    def _calculate_reward(self, metrics: Dict, action: int = None) -> float:
        """计算奖励"""
        reward = 0.0
        if metrics['packets_delivered'] > 0:
            reward += 1.0
        if metrics['delay'] > 0:
            reward -= metrics['delay'] * 0.1
        if metrics['packet_loss'] > 0:
            reward -= metrics['packet_loss'] * 0.5
        # 添加缓冲区使用率惩罚
        buffer_usage = metrics.get('buffer_usage', 0.0)
        reward -= buffer_usage * 0.3
        return reward
        
    def set_routing_algorithm(self, algorithm: RoutingAlgorithm):
        """设置路由算法"""
        self.routing_algorithm = algorithm 