from train_base import BaseTrainer
import torch

class PPOTrainer(BaseTrainer):
    def _run_episode(self, source, destination):
        path = [source]
        total_reward = 0
        current_leo = source
        
        metrics = {
            'delays': [],
            'bandwidths': [],
            'loss_rates': [],
            'queue_utilizations': []
        }
        
        while len(path) < MAX_PATH_LENGTH:
            state = self.agent.get_state(self.env, current_leo, destination)
            available_actions = self.env.get_available_actions(current_leo)
            
            if not available_actions:
                break
                
            action = self.agent.choose_action(state, available_actions,
                                            self.env, current_leo, destination, path)
            if action is None:
                break
                
            next_state, reward, done, info = self.env.step(current_leo, action, path)
            
            # 记录性能指标
            link_stats = info.get('link_stats', {})
            metrics['delays'].append(link_stats.get('delay', 0))
            metrics['bandwidths'].append(link_stats.get('bandwidth', 0))
            metrics['loss_rates'].append(link_stats.get('loss', 0))
            metrics['queue_utilizations'].append(link_stats.get('queue_utilization', 0))
            
            # 存储奖励和mask
            self.agent.rewards.append(reward)
            self.agent.masks.append(1 - done)
            
            total_reward += reward
            current_leo = list(self.env.leo_nodes.keys())[action]
            path.append(current_leo)
            
            if done or current_leo == destination:
                break
        
        # 在episode结束时更新策略
        with torch.no_grad():
            _, next_value = self.agent.actor_critic(
                torch.FloatTensor(next_state).to(self.agent.device)
            )
        self.agent.update(next_value)
        
        return {
            'path': path,
            'total_reward': total_reward,
            'metrics': metrics
        }
    
    def _save_checkpoint(self, episode, rewards, stats):
        torch.save({
            'episode': episode,
            'model_state_dict': self.agent.actor_critic.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'rewards': rewards,
            'performance_stats': dict(stats)
        }, f'models/ppo_checkpoint_episode_{episode+1}.pth')
    
    def _save_final_model(self):
        torch.save(self.agent.actor_critic.state_dict(), 'models/ppo_final_model.pth') 