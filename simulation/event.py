from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
from enum import Enum, auto

@dataclass
class Event:
    """事件基类"""
    time: float  # 事件发生时间
    type: 'EventType'  # 事件类型
    data: Optional[Dict[str, Any]] = None  # 事件数据
    priority: int = 0  # 事件优先级

    def __init__(self, event_type: 'EventType', time: float, data: Optional[Dict[str, Any]] = None, priority: int = 0):
        """
        初始化事件
        
        Args:
            event_type: 事件类型(EventType枚举)
            time: 事件发生时间
            data: 事件数据
            priority: 事件优先级
        """
        self.time = float(time)  # 确保时间是浮点数
        self.type = event_type
        self.data = data if data is not None else {}
        self.priority = priority
    
    def __lt__(self, other):
        """
        比较两个事件的优先级
        
        Args:
            other: 另一个事件
            
        Returns:
            bool: 当前事件是否优先级更低
        """
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.priority) < (other.time, other.priority)
    
    def __eq__(self, other):
        """
        比较两个事件是否相等
        
        Args:
            other: 另一个事件
            
        Returns:
            bool: 两个事件是否相等
        """
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.priority) == (other.time, other.priority)

class PacketEvent(Event):
    """数据包相关事件"""
    def __init__(self, time: float, event_type: str, packet_id: str, 
                 src: int, dst: int, current_node: int,
                 size: int, qos_class: int):
        super().__init__(
            time=time,
            type=event_type,
            data={
                'packet_id': packet_id,
                'src': src,
                'dst': dst,
                'current_node': current_node,
                'size': size,
                'qos_class': qos_class,
                'creation_time': time
            }
        )

class LinkEvent(Event):
    """链路相关事件"""
    def __init__(self, time: float, event_type: str,
                 src: int, dst: int, capacity: float, delay: float):
        super().__init__(
            time=time,
            type=event_type,
            data={
                'src': src,
                'dst': dst,
                'capacity': capacity,
                'delay': delay
            }
        )

class TopologyEvent(Event):
    """拓扑相关事件"""
    def __init__(self, time: float, event_type: str,
                 node_id: int, position: tuple, velocity: tuple):
        super().__init__(
            time=time,
            type=event_type,
            data={
                'node_id': node_id,
                'position': position,
                'velocity': velocity
            }
        )

# 事件类型枚举
class EventType(Enum):
    """事件类型枚举"""
    PACKET_GENERATION = auto()  # 数据包生成
    PACKET_ARRIVAL = auto()     # 数据包到达
    PACKET_SERVICE = auto()     # 数据包服务完成
    LINK_UPDATE = auto()        # 链路更新
    TOPOLOGY_UPDATE = auto()    # 拓扑更新

    # 数据包事件
    PACKET_DEPARTURE = "packet_departure"
    PACKET_DROP = "packet_drop"
    PACKET_TIMEOUT = "packet_timeout"
    
    # 链路事件
    LINK_UP = "link_up"
    LINK_DOWN = "link_down"
    
    # 拓扑事件
    NODE_FAILURE = "node_failure"
    NODE_RECOVERY = "node_recovery" 