import torch
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from ppo_model import PPOAgent
from mappo_model import MAPPOAgent
from config import *
import time
import argparse
import random
import numpy as np
from train import train_ppo, train_mappo

def set_random_seeds(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='卫星网络路由训练程序')
    parser.add_argument('--algo', type=str, default='ppo', choices=['dqn', 'ppo', 'mappo'],
                      help='选择要训练的算法 (dqn, ppo, 或 mappo)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='训练回合数')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    parser.add_argument('--n_agents', type=int, default=2,
                      help='MAPPO的智能体数量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 创建环境
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 打印使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    start_time = time.time()
    
    try:
        if args.algo == "ppo":
            print("\n开始训练 PPO 算法...")
            agent = PPOAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
            train_ppo(env, agent, num_episodes=args.episodes)
            
        elif args.algo == "mappo":
            print("\n开始训练 MAPPO 算法...")
            agent = MAPPOAgent(state_size, action_size, args.n_agents, 
                             env.get_leo_names(), env.get_leo_to_meo_mapping())
            train_mappo(env, agent, num_episodes=args.episodes)
            
        elif args.algo == "dqn":
            print("\n开始训练 DQN 算法...")
            agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
            # DQN的训练代码...
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        
    finally:
        end_time = time.time()
        training_time = end_time - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    main() 