from train import train
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
import torch
import random
import numpy as np

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
    # 设置随机种子
    set_random_seeds()
    
    # 打印训练设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 开始训练
        print("开始训练...")
        train()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
    finally:
        print("\n训练结束")

if __name__ == "__main__":
    main() 