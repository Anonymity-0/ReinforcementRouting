from typing import Dict, Any, List, Tuple
import numpy as np
from .packet import Packet

class TrafficGenerator:
    """流量生成器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化流量生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.packet_rate = config['simulation']['traffic']['poisson_lambda']
        self.next_packet_generation = 0.0
        self.total_satellites = config['simulation']['network']['total_satellites']
        self.packet_size = config['simulation']['traffic']['packet_size']
        
        # 预计算QoS类别权重
        self.qos_classes = config['simulation']['traffic']['qos_classes']
        self.qos_weights = [qos['weight'] for qos in self.qos_classes]
        
        # 缓存常用值
        self.satellite_indices = np.arange(self.total_satellites)
        
    def generate_packet(self, current_time: float) -> Packet:
        """
        生成新的数据包
        
        Args:
            current_time: 当前时间
            
        Returns:
            Packet: 生成的数据包
        """
        # 随机选择源节点和目标节点（使用numpy的高效操作）
        source, destination = self._select_source_destination()
        
        # 随机选择QoS类别
        qos_class = self._select_qos_class()
        
        # 生成数据包
        packet = Packet(
            source=source,
            destination=destination,
            size=self.packet_size,
            qos_class=qos_class
        )
        
        # 设置创建时间
        packet.create_time = current_time
        
        # 更新下一个数据包生成时间（使用向量化操作）
        self.next_packet_generation = current_time + self._generate_next_interval()
        
        return packet
        
    def generate_packets_batch(self, current_time: float, batch_size: int) -> List[Packet]:
        """
        批量生成数据包
        
        Args:
            current_time: 当前时间
            batch_size: 批量大小
            
        Returns:
            List[Packet]: 数据包列表
        """
        # 批量生成源节点和目标节点
        sources, destinations = self._select_source_destination_batch(batch_size)
        
        # 批量生成QoS类别
        qos_classes = self._select_qos_class_batch(batch_size)
        
        # 批量生成数据包
        packets = []
        for i in range(batch_size):
            packet = Packet(
                source=sources[i],
                destination=destinations[i],
                size=self.packet_size,
                qos_class=qos_classes[i]
            )
            packet.create_time = current_time
            packets.append(packet)
        
        # 更新下一个数据包生成时间
        self.next_packet_generation = current_time + self._generate_next_interval()
        
        return packets
    
    def _select_source_destination(self) -> Tuple[int, int]:
        """选择源节点和目标节点"""
        source = np.random.randint(0, self.total_satellites)
        destination = source
        while destination == source:
            destination = np.random.randint(0, self.total_satellites)
        return source, destination
    
    def _select_source_destination_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """批量选择源节点和目标节点"""
        sources = np.random.randint(0, self.total_satellites, size=batch_size)
        destinations = np.random.randint(0, self.total_satellites, size=batch_size)
        
        # 确保目标节点不是源节点
        mask = sources == destinations
        while np.any(mask):
            destinations[mask] = np.random.randint(0, self.total_satellites, size=np.sum(mask))
            mask = sources == destinations
            
        return sources, destinations
    
    def _select_qos_class(self) -> Dict:
        """选择QoS类别"""
        idx = np.random.choice(len(self.qos_classes), p=self.qos_weights)
        return self.qos_classes[idx]
    
    def _select_qos_class_batch(self, batch_size: int) -> List[Dict]:
        """批量选择QoS类别"""
        indices = np.random.choice(len(self.qos_classes), size=batch_size, p=self.qos_weights)
        return [self.qos_classes[idx] for idx in indices]
    
    def _generate_next_interval(self) -> float:
        """生成下一个数据包的时间间隔"""
        return np.random.exponential(1.0 / self.packet_rate) 