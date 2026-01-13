import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

from mmi_env import MMIOptEnv


class RewardLoggingCallback(BaseCallback):
    """
    自定义回调类，用于在训练过程中打印奖励和相关信息
    """
    def __init__(self, print_freq=100, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.print_freq = print_freq  # 每隔多少步打印一次
        self.episode_rewards = []
        self.episode_transmissions = []
        self.episode_imbalances = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
    def _on_step(self) -> bool:
        # 获取当前步的奖励和info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            reward = self.locals['rewards'][0]
            
            self.current_episode_reward += reward
            self.current_episode_steps += 1
            
            # 每隔 print_freq 步打印一次
            if self.num_timesteps % self.print_freq == 0:
                print(f"\n{'='*60}")
                print(f"时间步: {self.num_timesteps}")
                print(f"当前步奖励: {reward:.4f}")
                if 'transmission' in info:
                    print(f"  总传输: {info['transmission']:.4f}")
                    print(f"  上端口 (S21): {info['t_top']:.4f}")
                    print(f"  下端口 (S31): {info['t_bot']:.4f}")
                    print(f"  不平衡度: {info['imbalance']:.4f}")
                print(f"{'='*60}")
            
            # 检测episode结束
            done = self.locals.get('dones', [False])[0]
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                if 'transmission' in info:
                    self.episode_transmissions.append(info['transmission'])
                    self.episode_imbalances.append(info['imbalance'])
                
                # 打印episode总结
                print(f"\n{'#'*60}")
                print(f"Episode 完成!")
                print(f"  总奖励: {self.current_episode_reward:.4f}")
                print(f"  步数: {self.current_episode_steps}")
                if 'transmission' in info:
                    print(f"  最终总传输: {info['transmission']:.4f}")
                    print(f"  最终不平衡度: {info['imbalance']:.4f}")
                
                # 如果有多个episode，打印平均统计
                if len(self.episode_rewards) > 0:
                    print(f"\n最近 {len(self.episode_rewards)} 个Episodes统计:")
                    print(f"  平均奖励: {np.mean(self.episode_rewards):.4f} ± {np.std(self.episode_rewards):.4f}")
                    if len(self.episode_transmissions) > 0:
                        print(f"  平均传输: {np.mean(self.episode_transmissions):.4f} ± {np.std(self.episode_transmissions):.4f}")
                        print(f"  平均不平衡: {np.mean(self.episode_imbalances):.4f} ± {np.std(self.episode_imbalances):.4f}")
                print(f"{'#'*60}\n")
                
                # 重置episode计数
                self.current_episode_reward = 0
                self.current_episode_steps = 0
        
        return True


def train():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = MMIOptEnv()
    env = Monitor(env, log_dir)
    
    # Initialize Agent
    # PPO is a good default for continuous control
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 创建自定义回调
    reward_callback = RewardLoggingCallback(print_freq=100)  # 每100步打印一次
    
    print("Starting training...")
    print("将每隔100步打印奖励信息，每个episode结束时打印总结\n")
    
    # Train for 10,000 timesteps (~200 episodes of 50 steps)
    model.learn(total_timesteps=10000, callback=reward_callback)
    
    # Save the model
    model.save("mmi_ppo_model")
    print("Training complete. Model saved to mmi_ppo_model.zip")
    
    # Evaluate
    print("\n" + "="*60)
    print("评估训练好的模型...")
    print("="*60)
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 打印每一步的详细信息
        print(f"\nStep {i+1}:")
        print(f"  奖励: {reward:.4f}")
        print(f"  总传输: {info['transmission']:.4f}")
        print(f"  上端口 (S21): {info['t_top']:.4f}")
        print(f"  下端口 (S31): {info['t_bot']:.4f}")
        print(f"  不平衡度: {info['imbalance']:.4f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "="*60)        
    print(f"评估Episode总奖励: {total_reward:.4f}")
    print(f"最终信息: {info}")
    print("="*60)

if __name__ == "__main__":
    train()
