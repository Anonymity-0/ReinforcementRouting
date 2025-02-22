import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    """日志记录器类"""
    
    def __init__(self,
                save_dir: str,
                mode: str,
                name: str = 'satellite_network',
                level: int = logging.INFO):
        """
        初始化日志记录器
        
        Args:
            save_dir: 保存目录
            mode: 运行模式
            name: 日志记录器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建文件处理器
        log_file = os.path.join(save_dir, f'{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str) -> None:
        """
        记录INFO级别的日志
        
        Args:
            msg: 日志消息
        """
        self.logger.info(msg)
    
    def debug(self, msg: str) -> None:
        """
        记录DEBUG级别的日志
        
        Args:
            msg: 日志消息
        """
        self.logger.debug(msg)
    
    def warning(self, msg: str) -> None:
        """
        记录WARNING级别的日志
        
        Args:
            msg: 日志消息
        """
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """
        记录ERROR级别的日志
        
        Args:
            msg: 日志消息
        """
        self.logger.error(msg)
    
    def critical(self, msg: str) -> None:
        """
        记录CRITICAL级别的日志
        
        Args:
            msg: 日志消息
        """
        self.logger.critical(msg)
    
    def set_level(self, level: int) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level) 
 