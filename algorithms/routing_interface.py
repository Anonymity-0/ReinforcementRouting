from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class RoutingAlgorithm(ABC):
    """路由算法基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化路由算法
        
        Args:
            config: 配置字典
        """
        self.config = config
    
    @abstractmethod
    def get_next_hop(self, current_node: int, target_node: int, state: Dict[str, Any]) -> int:
        """
        获取下一跳节点
        
        Args:
            current_node: 当前节点
            target_node: 目标节点
            state: 状态信息
            
        Returns:
            int: 下一跳节点ID
        """
        pass
    
    @abstractmethod
    def update(self, 
              states: Dict[str, Any],
              actions: Any,
              rewards: Any,
              next_states: Dict[str, Any],
              dones: Any) -> Dict[str, float]:
        """
        更新算法
        
        Args:
            states: 状态字典
            actions: 动作
            rewards: 奖励
            next_states: 下一个状态字典
            dones: 结束标志
            
        Returns:
            Dict[str, float]: 训练信息
        """
        pass
    
    @abstractmethod
    def train(self) -> None:
        """设置为训练模式"""
        pass
    
    @abstractmethod
    def eval(self) -> None:
        """设置为评估模式"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        pass 