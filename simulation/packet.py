from typing import List, Dict, Any
import time

class Packet:
    """数据包类"""
    
    def __init__(self, source: int, destination: int, size: int, qos_class: Dict[str, Any]):
        """
        初始化数据包
        
        Args:
            source: 源节点ID
            destination: 目标节点ID
            size: 数据包大小(bytes)
            qos_class: QoS类别配置
        """
        self.source = source
        self.destination = destination
        self.size = size
        self.qos_class = qos_class
        
        # 路由信息
        self.current_node = source
        self.route = [source]
        self.is_delivered = False
        self.drop_reason = None  # 丢包原因
        
        # 时间信息(使用仿真时间)
        self.create_time = None  # 将在加入队列时设置
        self.deliver_time = None
        
    def update_route(self, next_hop: int, current_time: float) -> None:
        """
        更新路由路径
        
        Args:
            next_hop: 下一跳节点ID
            current_time: 当前仿真时间
        """
        self.current_node = next_hop
        self.route.append(next_hop)
        
        # 如果到达目的地
        if next_hop == self.destination:
            self.is_delivered = True
            self.deliver_time = current_time
            
    def get_delay(self) -> float:
        """
        获取传输延迟
        
        Returns:
            float: 传输延迟(秒)
        """
        if not self.create_time:
            return 0.0
            
        if self.is_delivered and self.deliver_time:
            return max(0.0, self.deliver_time - self.create_time)
        return 0.0  # 如果包未传递,返回0
        
    def get_route_length(self) -> int:
        """
        获取路由长度
        
        Returns:
            int: 路由长度(跳数)
        """
        return len(self.route) - 1  # 不包括源节点
        
    def __str__(self) -> str:
        """字符串表示"""
        status = "已送达" if self.is_delivered else "传输中"
        return (f"数据包[源节点:{self.source}, 目标节点:{self.destination}, "
                f"当前节点:{self.current_node}, 大小:{self.size}bytes, "
                f"QoS类别:{self.qos_class['name']}, 状态:{status}]") 