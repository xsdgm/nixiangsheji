import os
import sys
import numpy as np

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mmi_env import MMIOptEnv

def debug_validate():
    # Model path
    model_path = "mmi_ppo_model.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {os.path.abspath(model_path)}...")
    
    env = MMIOptEnv()
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Starting debug validation...\n")
    
    obs, info = env.reset()
    total_reward = 0
    
    print("="*80)
    print(f"{'Step':<5} | {'Action Mean':<12} | {'Action Std':<12} | {'Reward':<10} | {'Total T':<10} | {'Imbalance':<10}")
    print("="*80)
    
    # Run 10 steps for debugging
    for step in range(10):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Print action statistics
        action_mean = np.mean(action)
        action_std = np.std(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        t_tot = info.get('transmission', 0)
        imb = info.get('imbalance', 0)
        
        print(f"{step+1:<5} | {action_mean:<12.6f} | {action_std:<12.6f} | {reward:<10.4f} | {t_tot:<10.4f} | {imb:<10.4f}")
        
        # Print geometry change
        if step == 0:
            initial_density = env.geo.get_density()
        elif step % 3 == 0:
            current_density = env.geo.get_density()
            diff = np.sum(np.abs(current_density - initial_density))
            print(f"  -> Geometry change from initial: {diff:.2f} pixels")
        
        if terminated or truncated:
            break
    
    print("="*80)
    print(f"\nDebug Summary:")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Final Transmission: {t_tot:.4f}")
    print(f"  Final Imbalance: {imb:.4f}")

if __name__ == "__main__":
    debug_validate()
