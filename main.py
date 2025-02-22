# 导入必要的库
import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
import torch  # 添加torch导入
from simulation.simulator import NetworkSimulator
from algorithms.dijkstra_algorithm import DijkstraAlgorithm
from algorithms.ppo_algorithm import PPOAlgorithm
from algorithms.mappo_algorithm import MAPPOAlgorithm
from algorithms.ospf_algorithm import OSPFAlgorithm
from experiments.run_experiments import train_ppo, evaluate_ppo, run_dijkstra, train_mappo, evaluate_mappo, run_ospf


def setup_logger():
    """设置日志记录器"""
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    
    return logger

def get_device():
    """获取可用的计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        return device, f"GPU ({device_name})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, "Apple M系列GPU (MPS)"
    else:
        device = torch.device("cpu")
        return device, "CPU"

def main():
    """主函数"""
    # 获取计算设备
    device, device_name = get_device()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='卫星网络路由训练程序')
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'eval', 'dijkstra', 'ospf'],
                      help='运行模式: train、eval、dijkstra或ospf')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                      choices=['ppo', 'mappo'],
                      help='算法选择: ppo或mappo（仅在train/eval模式下使用）')
    parser.add_argument('--model_path', type=str, default=None,
                      help='模型路径(仅在eval模式下使用)')
    args = parser.parse_args()
    
    # 加载配置文件
    with open('config/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    with open('config/tle_config.yaml', 'r') as f:
        tle_config = yaml.safe_load(f)
    
    with open('config/link_config.yaml', 'r') as f:
        link_config = yaml.safe_load(f)
        
    with open('config/algorithm_config.yaml', 'r') as f:
        algorithm_config = yaml.safe_load(f)
    
    # 设置计算设备
    algorithm_config['common']['device'] = str(device)
    
    # 合并配置
    config = {
        'simulation': {
            'common': base_config['simulation']['common'],
            'network': {
                **base_config['simulation']['network'],
                'total_satellites': tle_config['topology']['total_satellites']
            },
            'traffic': base_config['simulation']['traffic'],
            'topology': tle_config['topology'],
            'link': link_config['link']
        },
        'algorithm': algorithm_config,
        'logging': base_config['logging'],
        'visualization': base_config['visualization']
    }
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('results', f'{args.mode}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger()
    
    # 显示运行信息
    logger.info("="*50)
    logger.info(f"运行模式: {args.mode}")
    if args.mode in ['train', 'eval']:
        logger.info(f"算法: {args.algorithm.upper()}")
    logger.info(f"计算设备: {device_name}")
    logger.info("="*50)
    
    # 创建环境
    env = NetworkSimulator(config)
    
    if args.mode == 'train':
        # 训练模式
        logger.info("开始训练...")
        if args.algorithm == 'ppo':
            algorithm = train_ppo(env, save_dir, logger)
            # 评估训练后的模型
            evaluate_ppo(env, os.path.join(save_dir, 'ppo_model_final.pth'), logger, num_episodes=10)
        elif args.algorithm == 'mappo':
            algorithm = train_mappo(env, save_dir, logger)
            # 评估训练后的模型
            evaluate_mappo(env, os.path.join(save_dir, 'mappo_model_final.pth'), logger, num_episodes=10)
        else:
            raise ValueError(f"不支持的算法: {args.algorithm}")
        logger.info("训练完成!")
    elif args.mode == 'eval':
        # 评估模式
        if args.model_path is None:
            raise ValueError("在eval模式下必须指定model_path!")
            
        logger.info(f"开始评估 {args.algorithm.upper()} 模型: {args.model_path}")
        if args.algorithm == 'ppo':
            evaluate_ppo(env, args.model_path, logger)
        elif args.algorithm == 'mappo':
            evaluate_mappo(env, args.model_path, logger)
        logger.info("评估完成!")
    elif args.mode == 'dijkstra':
        # Dijkstra模式
        logger.info("开始运行Dijkstra算法...")
        run_dijkstra(env, save_dir, logger)
        logger.info("Dijkstra算法运行完成!")
    elif args.mode == 'ospf':
        # OSPF模式
        logger.info("开始运行OSPF算法...")
        run_ospf(env, save_dir, logger)
        logger.info("OSPF算法运行完成!")

if __name__ == '__main__':
    main() 