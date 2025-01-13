import torch
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from ppo_model import PPOAgent
from mappo_model import MAPPOAgent
from config import *
import time
from train import train_ppo, train_mappo

def main():
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    
    # 创建环境
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 打印使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 训练选定的算法
    algorithm = "ppo"  # 可以选择 "dqn", "ppo", 或 "mappo"
    
    start_time = time.time()
    
    try:
        if algorithm == "ppo":
            print("\n开始训练 PPO 算法...")
            agent = PPOAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
            train_ppo(env, agent)
            
        elif algorithm == "mappo":
            print("\n开始训练 MAPPO 算法...")
            n_agents = 2  # 设置智能体数量
            agent = MAPPOAgent(state_size, action_size, n_agents, env.get_leo_names(), env.get_leo_to_meo_mapping())
            train_mappo(env, agent)
            
        elif algorithm == "dqn":
            print("\n开始训练 DQN 算法...")
            agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
            # DQN的训练代码...
            
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