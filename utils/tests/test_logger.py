import unittest
import os
import shutil
import logging
from ..logger import Logger

class TestLogger(unittest.TestCase):
    """日志记录器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试目录
        cls.test_dir = 'test_logs'
        os.makedirs(cls.test_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 删除测试目录
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_init(self):
        """测试初始化"""
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        logger = Logger(self.test_dir, 'test')
        
        # 检查日志记录器属性
        self.assertEqual(logger.logger.name, 'satellite_network')
        self.assertEqual(logger.logger.level, logging.INFO)
        self.assertEqual(len(logger.logger.handlers), 2)  # 文件处理器和控制台处理器
        
        # 检查处理器属性
        for handler in logger.logger.handlers:
            self.assertEqual(handler.level, logging.INFO)
            self.assertIsNotNone(handler.formatter)
    
    def test_log_levels(self):
        """测试不同级别的日志记录"""
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        logger = Logger(self.test_dir, 'test')
        
        # 记录不同级别的日志
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # 检查日志文件是否生成
        log_files = [f for f in os.listdir(self.test_dir) if f.endswith('.log')]
        self.assertEqual(len(log_files), 1)
        
        # 检查日志文件内容
        with open(os.path.join(self.test_dir, log_files[0]), 'r') as f:
            content = f.read()
            # DEBUG级别的消息不应该被记录（默认级别是INFO）
            self.assertNotIn('Debug message', content)
            self.assertIn('Info message', content)
            self.assertIn('Warning message', content)
            self.assertIn('Error message', content)
            self.assertIn('Critical message', content)
    
    def test_set_level(self):
        """测试设置日志级别"""
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        logger = Logger(self.test_dir, 'test')
        
        # 设置为DEBUG级别
        logger.set_level(logging.DEBUG)
        self.assertEqual(logger.logger.level, logging.DEBUG)
        for handler in logger.logger.handlers:
            self.assertEqual(handler.level, logging.DEBUG)
        
        # 记录DEBUG消息
        logger.debug('Debug message')
        
        # 检查日志文件内容
        log_files = [f for f in os.listdir(self.test_dir) if f.endswith('.log')]
        with open(os.path.join(self.test_dir, log_files[0]), 'r') as f:
            content = f.read()
            self.assertIn('Debug message', content)
    
    def test_multiple_loggers(self):
        """测试多个日志记录器"""
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        logger1 = Logger(self.test_dir, 'test1', name='logger1')
        logger2 = Logger(self.test_dir, 'test2', name='logger2')
        
        # 记录日志
        logger1.info('Message from logger1')
        logger2.info('Message from logger2')
        
        # 检查日志文件
        log_files = [f for f in os.listdir(self.test_dir) if f.endswith('.log')]
        self.assertEqual(len(log_files), 2)
        
        # 检查每个日志文件的内容
        for log_file in log_files:
            with open(os.path.join(self.test_dir, log_file), 'r') as f:
                content = f.read()
                if 'test1' in log_file:
                    self.assertIn('Message from logger1', content)
                    self.assertNotIn('Message from logger2', content)
                else:
                    self.assertIn('Message from logger2', content)
                    self.assertNotIn('Message from logger1', content)

if __name__ == '__main__':
    unittest.main() 
 