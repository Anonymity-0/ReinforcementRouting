import torch
import numpy as np
from satellite_env import SatelliteEnv
from dqn_model import DQNAgent
from config import *
import random
import time
from collections import deque
import matplotlib.pyplot as plt

def train_dqn():
    """训练DQN代理"""
    print("初始化环境...")
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    print(f"状态空间大小: {state_size}")
    print(f"动作空间大小: {action_size}")
    
    # 初始化DQN代理
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    
    # 训练统计
    episode_rewards = []
    avg_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    print("\n开始训练...")
    try:
        for episode in range(NUM_EPISODES):
            total_reward = 0
            state_size, action_size = env.reset()
            
            # 随机选择源节点和目标节点
            all_leos = env.get_leo_names()
            source = random.choice(all_leos)
            destination = random.choice([leo for leo in all_leos if leo != source])
            
            path = [source]
            current_leo = source
            step = 0
            
            while step < MAX_PATH_LENGTH:
                # 获取当前状态
                state = agent.get_state(env, current_leo, destination)
                
                # 获取可用动作
                available_actions = env.get_available_actions(current_leo)
                
                # 选择动作
                action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
                
                if action is None:
                    break
                    
                # 执行动作
                next_leo = env.get_leo_names()[action]
                next_state, reward, done, info = env.step(current_leo, action, path)
                
                # 存储经验
                agent.memorize(state, action, reward, next_state, done)
                
                # 经验回放
                if len(agent.memory) > BATCH_SIZE:
                    agent.replay(BATCH_SIZE)
                
                total_reward += reward
                current_leo = next_leo
                path.append(current_leo)
                
                if done or current_leo == destination:
                    break
                    
                step += 1
            
            # 更新目标网络
            if episode % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            # 记录统计信息
            episode_rewards.append(total_reward)
            avg_rewards.append(total_reward)
            avg_reward = np.mean(avg_rewards)
            
            # 打印训练进度
            if episode % 10 == 0:
                print(f"\nEpisode {episode}/{NUM_EPISODES}")
                print(f"源节点: {source} -> 目标节点: {destination}")
                print(f"路径: {' -> '.join(path)}")
                print(f"总奖励: {total_reward:.2f}")
                print(f"平均奖励: {avg_reward:.2f}")
                print(f"探索率: {agent.epsilon:.4f}")
                
                # 如果性能提升，保存模型
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(agent.policy_net.state_dict(), 'best_model.pth')
                    print(f"保存最佳模型，平均奖励: {best_avg_reward:.2f}")
        
        # 训练结束后绘制奖励曲线
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('训练奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('training_rewards.png')
        plt.close()
        
        print("\n训练完成!")
        print(f"最佳平均奖励: {best_avg_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
    finally:
        # 保存最终模型
        torch.save(agent.policy_net.state_dict(), 'final_model.pth')
        print("已保存最终模型")

def evaluate_model(model_path, num_episodes=100):
    """评估训练好的模型"""
    print("\n开始评估模型...")
    
    env = SatelliteEnv()
    state_size, action_size = env.reset()
    
    # 初始化代理并加载模型
    agent = DQNAgent(state_size, action_size, env.get_leo_names(), env.get_leo_to_meo_mapping())
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.epsilon = 0.0  # 关闭探索
    
    success_count = 0
    total_rewards = []
    path_lengths = []
    
    for episode in range(num_episodes):
        state_size, action_size = env.reset()
        
        # 随机选择源节点和目标节点
        all_leos = env.get_leo_names()
        source = random.choice(all_leos)
        destination = random.choice([leo for leo in all_leos if leo != source])
        
        path = [source]
        current_leo = source
        total_reward = 0
        step = 0
        
        while step < MAX_PATH_LENGTH:
            state = agent.get_state(env, current_leo, destination)
            available_actions = env.get_available_actions(current_leo)
            action = agent.choose_action(state, available_actions, env, current_leo, destination, path)
            
            if action is None:
                break
                
            next_leo = env.get_leo_names()[action]
            next_state, reward, done, info = env.step(current_leo, action, path)
            
            total_reward += reward
            current_leo = next_leo
            path.append(current_leo)
            
            if done or current_leo == destination:
                if current_leo == destination:
                    success_count += 1
                break
                
            step += 1
        
        total_rewards.append(total_reward)
        path_lengths.append(len(path))
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"路径: {' -> '.join(path)}")
            print(f"路径长度: {len(path)}")
            print(f"总奖励: {total_reward:.2f}")
    
    # 打印评估结果
    success_rate = success_count / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_path_length = np.mean(path_lengths)
    
    print("\n评估结果:")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均路径长度: {avg_path_length:.2f}")

def train():
    """训练入口函数"""
    start_time = time.time()
    
    try:
        # 训练模型
        train_dqn()
        
        # 评估最佳模型
        evaluate_model('best_model.pth')
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
    finally:
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\n总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")

if __name__ == "__main__":
    train()