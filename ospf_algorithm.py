import numpy as np
from typing import Dict, List, Any, Tuple
from .routing_interface import RoutingAlgorithm

class OSPFAlgorithm(RoutingAlgorithm):
    """OSPF路由算法实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化OSPF算法
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.total_satellites = config['simulation']['network']['total_satellites']
        self.routing_table = {}  # 路由表
        self.link_state_database = {}  # 链路状态数据库
        self.max_queue_length = config['simulation']['network'].get('max_queue_length', 1000)
        self.bandwidth = config['simulation']['link'].get('bandwidth', 1e9)  # 默认1Gbps
        
    def _calculate_link_cost(self, src: int, dst: int, state: Dict) -> float:
        """
        计算链路开销
        
        Args:
            src: 源节点
            dst: 目标节点
            state: 状态字典
            
        Returns:
            float: 链路开销
        """
        topology = state['topology']['object']
        link_data = topology.get_link_data(src, dst)
        
        # 基础开销（传播延迟）
        base_cost = link_data['delay'] * 1000  # 转换为毫秒
        
        # 带宽因子
        bandwidth_factor = self.bandwidth / (link_data['capacity'] + 1e-6)  # 避免除零
        
        # 链路质量因子
        quality_factor = 1 / (link_data['quality'] + 1e-6)  # 避免除零
        
        # 队列长度因子
        queue_lengths = state['network']['queue_lengths']
        queue_factor = 1 + queue_lengths[dst] / self.max_queue_length
        
        # 综合开销
        total_cost = base_cost * (0.4 * bandwidth_factor + 0.3 * quality_factor + 0.3 * queue_factor)
        
        return max(0.001, total_cost)  # 确保开销为正
        
    def _update_link_state_database(self, state: Dict) -> None:
        """
        更新链路状态数据库
        
        Args:
            state: 状态字典
        """
        topology = state['topology']['object']
        network_state = state['network']
        self.link_state_database = {}
        
        # 遍历所有节点对
        for i in range(self.total_satellites):
            self.link_state_database[i] = {}
            for j in range(self.total_satellites):
                if i != j:
                    # 检查链路状态
                    link_states = network_state['link_states']
                    if link_states[i][j] > 0:  # 如果链路可用
                        cost = self._calculate_link_cost(i, j, state)
                        self.link_state_database[i][j] = cost
                    
    def _run_dijkstra(self, source: int) -> Dict[int, Tuple[float, int]]:
        """
        运行Dijkstra算法计算最短路径
        
        Args:
            source: 源节点
            
        Returns:
            Dict[int, Tuple[float, int]]: 到各节点的最短距离和下一跳
        """
        distances = {source: 0}  # 距离
        next_hops = {source: None}  # 下一跳
        unvisited = set(range(self.total_satellites))
        
        while unvisited:
            # 找到未访问节点中距离最小的
            current = None
            min_dist = float('inf')
            for node in unvisited:
                if node in distances and distances[node] < min_dist:
                    current = node
                    min_dist = distances[node]
                    
            if current is None:
                break
                
            unvisited.remove(current)
            
            # 更新邻居的距离
            if current in self.link_state_database:
                for neighbor, cost in self.link_state_database[current].items():
                    if neighbor in unvisited:
                        new_dist = distances[current] + cost
                        if neighbor not in distances or new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            # 如果是源节点的直接邻居，记录为下一跳
                            if current == source:
                                next_hops[neighbor] = neighbor
                            # 否则继承当前节点的下一跳
                            else:
                                next_hops[neighbor] = next_hops[current]
                                
        return {node: (distances.get(node, float('inf')), next_hops.get(node)) 
                for node in range(self.total_satellites)}
        
    def _update_routing_table(self, state: Dict) -> None:
        """
        更新路由表
        
        Args:
            state: 状态字典
        """
        # 更新链路状态数据库
        self._update_link_state_database(state)
        
        # 为每个节点计算最短路径
        self.routing_table = {}
        for source in range(self.total_satellites):
            self.routing_table[source] = self._run_dijkstra(source)
            
    def get_next_hop(self, current_node: int, destination: int, state: Dict) -> int:
        """
        获取下一跳节点
        
        Args:
            current_node: 当前节点
            destination: 目标节点
            state: 状态字典
            
        Returns:
            int: 下一跳节点ID，如果无法到达则返回-1
        """
        # 如果已到达目标节点
        if current_node == destination:
            return destination
            
        # 更新路由表
        self._update_routing_table(state)
        
        # 获取路由信息
        if current_node not in self.routing_table or destination not in self.routing_table[current_node]:
            return -1
            
        distance, next_hop = self.routing_table[current_node][destination]
        
        # 如果没有可用路径
        if distance == float('inf') or next_hop is None:
            return -1
            
        # 检查链路状态
        network_state = state['network']
        link_states = network_state['link_states']
        
        # 如果下一跳链路不可用，返回-1
        if link_states[current_node][next_hop] <= 0:
            return -1
            
        return next_hop
        
    def update(self, states: Dict[str, Any], actions: Any, rewards: Any,
             next_states: Dict[str, Any], dones: Any) -> Dict[str, float]:
        """
        更新算法（OSPF是静态路由算法，不需要更新）
        """
        return {}
        
    def train(self) -> None:
        """设置为训练模式（OSPF不需要训练）"""
        pass
        
    def eval(self) -> None:
        """设置为评估模式（OSPF不需要训练）"""
        pass
        
    def save(self, path: str) -> None:
        """保存模型（OSPF不需要保存模型）"""
        pass
        
    def load(self, path: str) -> None:
        """加载模型（OSPF不需要加载模型）"""
        pass 