import numpy as np
from typing import Dict, List, Any, Tuple
from .routing_interface import RoutingAlgorithm

class DijkstraAlgorithm(RoutingAlgorithm):
    """Dijkstra路由算法实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Dijkstra算法
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.total_satellites = config['total_satellites']
        
    def _calculate_link_weight(self, src: int, dst: int, state: Dict) -> float:
        """
        计算链路权重
        
        Args:
            src: 源节点
            dst: 目标节点
            state: 状态字典
            
        Returns:
            float: 链路权重
        """
        topology = state['topology']['object']
        link_data = topology.get_link_data(src, dst)
        
        # 如果链路完全不可用才返回无穷大
        if link_data['capacity'] <= 0:
            return float('inf')
            
        # 计算基础权重（传播延迟）
        base_weight = link_data['delay'] * 1000  # 转换为毫秒
        
        # 考虑链路质量（降低影响）
        quality_factor = 0.2  # 降低质量惩罚系数
        quality_penalty = quality_factor * (1 - link_data['quality']) * base_weight
        
        # 考虑链路容量
        capacity_factor = 0.1
        capacity_penalty = capacity_factor * (1e9 / (link_data['capacity'] + 1e-6))  # 避免除零
        
        # 计算总权重
        total_weight = base_weight + quality_penalty + capacity_penalty
        
        return max(0.001, total_weight)  # 确保权重为正
        
    def get_next_hop(self, current_node: int, destination: int, state: Dict) -> int:
        """
        获取下一跳节点
        
        Args:
            current_node: 当前节点
            destination: 目标节点
            state: 状态字典
            
        Returns:
            int: 下一跳节点ID
                返回值定义:
                - 返回正整数: 表示下一跳节点ID
                - 返回 0: 表示已到达目的地
                - 返回 -1: 表示找不到有效路径
        """
        # 如果已到达目标节点
        if current_node == destination:
            return 0
            
        topology = state['topology']['object']
        
        # 初始化距离和访问标记
        distances = {current_node: 0}  # 使用字典存储距离
        previous = {current_node: None}  # 使用字典存储前驱节点
        unvisited = set(range(self.total_satellites))  # 未访问节点集合
        
        while unvisited:
            # 找到未访问节点中距离最小的
            current = None
            min_dist = float('inf')
            for node in unvisited:
                if node in distances and distances[node] < min_dist:
                    current = node
                    min_dist = distances[node]
            
            if current is None or current == destination:
                break
                
            # 从未访问集合中移除当前节点
            unvisited.remove(current)
            
            # 获取当前节点的邻居
            neighbors = topology.get_valid_neighbors(current)
            
            # 更新到邻居的距离
            for neighbor in neighbors:
                if neighbor not in unvisited:
                    continue
                    
                # 计算链路权重
                weight = self._calculate_link_weight(current, neighbor, state)
                if weight == float('inf'):
                    continue
                    
                # 更新距离
                new_distance = distances[current] + weight
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
        
        # 如果找不到到目标节点的路径
        if destination not in distances:
            return -1
            
        # 回溯找到第一跳节点
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        # 获取第一跳节点
        if len(path) < 2:
            return -1
            
        return path[1]  # path[0]是current_node
    
    def update(self, states: List[Dict], actions: np.ndarray, rewards: np.ndarray,
             next_states: List[Dict], dones: np.ndarray) -> None:
        """Dijkstra算法不需要更新"""
        pass
        
    def clear_buffer(self) -> None:
        """Dijkstra算法不需要缓冲区"""
        pass
        
    def train(self) -> None:
        """Dijkstra算法不需要训练"""
        pass
        
    def eval(self) -> None:
        """Dijkstra算法不需要评估模式"""
        pass
    
    def save(self, path: str) -> None:
        """Dijkstra算法不需要保存模型"""
        pass
    
    def load(self, path: str) -> None:
        """Dijkstra算法不需要加载模型"""
        pass 