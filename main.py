from train import train_dqn
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
import torch
import random
import numpy as np
import time

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
    
    start_time = time.time()
    try:
        # 训练模型
        train_dqn()
        
        # 评估最终模型
        evaluate_model('models/final_model.pth')
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
    finally:
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    main() 