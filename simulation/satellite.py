from typing import List, Dict, Any, Optional
from collections import deque
from .packet import Packet

class Satellite:
    """卫星节点类"""
    
    def __init__(self, satellite_id: int, max_queue_length: int, config: Dict[str, Any] = None):
        """
        初始化卫星节点
        
        Args:
            satellite_id: 卫星ID
            max_queue_length: 最大队列长度
            config: 配置参数
        """
        self.id = satellite_id
        self.max_queue_length = max_queue_length
        self.config = config or {
            'simulation': {
                'traffic': {
                    'poisson_lambda': 500.0,  # 默认每秒生成500个数据包
                    'packet_size': 1024  # 默认包大小1KB
                },
                'link': {
                    'bandwidth': 10e6  # 默认带宽10Mbps
                },
                'network': {
                    'total_satellites': 66  # Iridium星座的卫星总数
                }
            }
        }
        
        # 计算服务率（每秒可以处理的数据包数）
        packet_size_bits = self.config['simulation']['traffic']['packet_size'] * 8
        bandwidth_bps = self.config['simulation']['link']['bandwidth']
        self.service_rate = bandwidth_bps / packet_size_bits  # 每秒可处理的数据包数
        
        # 计算到达率（从泊松分布参数获取）
        self.arrival_rate = self.config['simulation']['traffic']['poisson_lambda'] / self.config['simulation']['network']['total_satellites']
        
        # 计算利用率 ρ = λ/μ
        self.utilization = self.arrival_rate / self.service_rate if self.service_rate > 0 else 0.0
        
        self.queue = []  # 数据包队列
        self.dropped_packets_history = []  # 丢弃的数据包历史记录
        
        # 性能统计
        self.total_packets = 0  # 总数据包数
        self.delivered_packets = 0  # 成功传输的数据包数
        self.lost_packets = 0  # 丢失的数据包数
        self.dropped_packets = 0  # 因缓冲区溢出丢弃的数据包数
        self.total_bytes_delivered = 0  # 成功传输的总字节数
        self.total_bytes_received = 0  # 接收到的总字节数
        self.packet_delays = []  # 数据包延迟列表
        
        # 时间统计
        self.busy_time = 0.0  # 卫星忙时间
        self.last_busy_start = 0.0  # 上次开始忙的时间
        self.queue_lengths = []  # 队列长度历史
        self.queue_length_times = []  # 队列长度对应的时间点
        
    def enqueue_packet(self, packet: 'Packet') -> bool:
        """
        将数据包加入队列
        
        Args:
            packet: 数据包
            
        Returns:
            bool: 是否成功加入队列
        """
        if len(self.queue) < self.max_queue_length:
            self.queue.append(packet)
            
            # 记录队列长度
            current_time = packet.create_time
            self.queue_lengths.append(len(self.queue))
            self.queue_length_times.append(current_time)
            
            # 如果队列从空变为非空，记录开始忙的时间
            if len(self.queue) == 1:
                self.last_busy_start = current_time
                
            return True
        else:
            self.dropped_packets += 1
            self.dropped_packets_history.append(packet)
            return False
            
    def dequeue_packet(self) -> Optional['Packet']:
        """
        从队列中取出数据包
        
        Returns:
            Optional[Packet]: 数据包，如果队列为空则返回None
        """
        if self.queue:
            packet = self.queue.pop(0)
            
            # 记录队列长度
            current_time = packet.deliver_time if packet.is_delivered else packet.create_time
            self.queue_lengths.append(len(self.queue))
            self.queue_length_times.append(current_time)
            
            # 如果队列变为空，更新忙时间
            if len(self.queue) == 0:
                self.busy_time += current_time - self.last_busy_start
                
            return packet
        return None
        
    def get_average_queue_length(self, current_time: float) -> float:
        """
        获取平均队列长度
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 平均队列长度
        """
        if not self.queue_lengths:
            return 0.0
            
        total_time = current_time - self.queue_length_times[0]
        if total_time <= 0:
            return max(0.0, float(self.queue_lengths[-1]))
            
        weighted_sum = 0.0
        for i in range(len(self.queue_lengths) - 1):
            # 确保队列长度非负
            queue_length = max(0, self.queue_lengths[i])
            time_diff = max(0.0, self.queue_length_times[i+1] - self.queue_length_times[i])
            weighted_sum += queue_length * time_diff
            
        # 加上最后一段时间
        last_queue_length = max(0, self.queue_lengths[-1])
        last_time_diff = max(0.0, current_time - self.queue_length_times[-1])
        weighted_sum += last_queue_length * last_time_diff
        
        # 确保返回值非负
        return max(0.0, weighted_sum / total_time)
        
    def get_utilization(self, current_time: float) -> float:
        """
        获取卫星利用率
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 利用率 [0,1]
        """
        if current_time <= 0:
            return 0.0
            
        # 假设链路带宽为1Gbps
        bandwidth = 1e9  # bits per second
            
        # 计算总的忙时间
        total_busy_time = 0.0
        
        # 计算所有已传输数据包的传输时间
        if self.total_bytes_delivered > 0:
            transmission_time = (self.total_bytes_delivered * 8) / bandwidth
            total_busy_time += transmission_time
            
        # 如果当前队列非空，加上当前数据包的传输时间
        if len(self.queue) > 0:
            current_packet = self.queue[0]
            transmission_time = (current_packet.size * 8) / bandwidth
            total_busy_time += transmission_time
            
        # 确保利用率不超过1
        return min(1.0, max(0.0, total_busy_time / current_time))
        
    def get_packet_delays(self) -> List[float]:
        """
        获取数据包延迟列表
        
        Returns:
            List[float]: 延迟列表
        """
        return self.packet_delays
        
    def get_delivery_rate(self) -> float:
        """
        获取数据包交付率
        
        Returns:
            float: 交付率 [0,1]
        """
        if self.total_packets == 0:
            return 0.0
        return self.delivered_packets / self.total_packets
        
    def get_throughput(self, current_time: float) -> float:
        """
        获取吞吐量(bps)
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 吞吐量(bps)
        """
        if current_time <= 0:
            return 0.0
        return (self.total_bytes_delivered * 8) / current_time
        
    def reset(self):
        """重置卫星状态"""
        self.queue.clear()
        self.total_packets = 0
        self.delivered_packets = 0
        self.lost_packets = 0
        self.dropped_packets = 0
        self.total_bytes_delivered = 0
        self.total_bytes_received = 0
        self.packet_delays.clear()
        self.busy_time = 0.0
        self.last_busy_start = 0.0
        self.queue_lengths.clear()
        self.queue_length_times.clear()
        
    def get_current_queue_length(self) -> float:
        """获取当前队列长度"""
        return len(self.queue)
        
    def get_current_utilization(self) -> float:
        """获取当前链路利用率"""
        if not self.queue_lengths:
            return 0.0
        
        # 获取最近的传输记录
        latest_transmission = self.queue_lengths[-1]
        return latest_transmission / self.max_queue_length
        
    def get_transmission_history(self) -> List[Dict[str, Any]]:
        """获取传输历史记录"""
        history = []
        for i in range(len(self.queue_lengths) - 1):
            history.append({
                'time': self.queue_length_times[i+1],
                'queue_length': self.queue_lengths[i],
                'capacity': self.max_queue_length
            })
        return history 