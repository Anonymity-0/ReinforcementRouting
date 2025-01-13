from train import train_and_evaluate
import torch
import random
import numpy as np
import time
import argparse

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
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo', 'mappo'],
                      help='选择要训练的算法 (dqn, ppo, 或 mappo)')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='训练回合数')
    parser.add_argument('--eval_episodes', type=int, default=100,
                      help='评估回合数')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 打印训练设备信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    start_time = time.time()
    try:
        # 训练并评估选定的算法
        print(f"\n开始训练 {args.algo.upper()} 算法...")
        train_stats, eval_stats = train_and_evaluate(
            algo_name=args.algo,
            train_episodes=args.episodes,
            eval_episodes=args.eval_episodes
        )
        
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