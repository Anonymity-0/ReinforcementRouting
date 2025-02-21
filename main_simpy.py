import os
import sys
import yaml
import logging
from datetime import datetime
from simulation_simpy.simulator import SimpyNetworkSimulator
from algorithms.dijkstra_algorithm import DijkstraAlgorithm
from algorithms.ppo_algorithm import PPOAlgorithm
from experiments.run_experiments import train_ppo, evaluate_ppo, run_dijkstra

def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

def main():
    """主函数"""
    # 设置日志记录器
    logger = setup_logger()
    
    # 加载配置文件
    logger.info("加载配置文件...")
    
    # 加载基础配置
    logger.info("基础配置: config/base_config.yaml")
    with open('config/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 加载TLE配置
    logger.info("TLE配置: config/tle_config.yaml")
    with open('config/tle_config.yaml', 'r') as f:
        tle_config = yaml.safe_load(f)
    
    # 加载链路配置
    logger.info("链路配置: config/link_config.yaml")
    with open('config/link_config.yaml', 'r') as f:
        link_config = yaml.safe_load(f)
    
    # 加载算法配置
    logger.info("算法配置: config/algorithm_config.yaml")
    with open('config/algorithm_config.yaml', 'r') as f:
        algorithm_config = yaml.safe_load(f)
    
    # 合并配置
    config = {
        'simulation': {
            'common': base_config['simulation']['common'],
            'network': {
                **base_config['simulation']['network'],
                'total_satellites': tle_config['topology']['total_satellites']
            },
            'link': link_config['link'],
            'traffic': base_config['simulation']['traffic'],
            'topology': tle_config['topology']
        },
        'algorithm': algorithm_config,
        'logging': base_config['logging'],
        'visualization': base_config['visualization']
    }
    
    # 创建保存目录
    save_dir = os.path.join('results_simpy', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境
    env = SimpyNetworkSimulator(config)
    
    # 根据模式选择操作
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dijkstra'
    if mode == '--mode':
        mode = sys.argv[2]
    
    if mode == 'train':
        logger.info("开始训练PPO算法(SimPy版本)...")
        train_ppo(env, save_dir, logger)
    elif mode == 'eval':
        logger.info("开始评估PPO算法(SimPy版本)...")
        evaluate_ppo(env, save_dir, logger)
    elif mode == 'dijkstra':
        logger.info("开始运行Dijkstra算法(SimPy版本)...")
        run_dijkstra(env, save_dir, logger)
    else:
        logger.error(f"不支持的模式: {mode}")

if __name__ == '__main__':
    main() 