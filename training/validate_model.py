import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mmi_env import MMIOptEnv

def validate():
    # Model path - checking parent directory as typically found there
    model_path = "../mmi_ppo_model.zip"
    if not os.path.exists(model_path):
        model_path = "mmi_ppo_model.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at ../mmi_ppo_model.zip or ./mmi_ppo_model.zip")
        return

    print(f"Loading model from {os.path.abspath(model_path)}...")
    
    env = MMIOptEnv()
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Starting validation...")
    
    # Run a few episodes to get a stable idea of performance if needed, 
    # but request implies "validate model" which often means one run or a summary.
    # We will run 1 episode.
    
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    step = 0
    MAX_STEPS = 50 
    
    print("-" * 80)
    print(f"{'Step':<5} | {'Reward':<10} | {'Total T':<10} | {'T Top':<10} | {'T Bot':<10} | {'Imbalance':<10}")
    print("-" * 80)
    
    t_tot = 0
    imb = 0
    
    while not done and step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        t_tot = info.get('transmission', 0)
        t_top = info.get('t_top', 0)
        t_bot = info.get('t_bot', 0)
        imb = info.get('imbalance', 0)
        
        print(f"{step:<5} | {reward:<10.4f} | {t_tot:<10.4f} | {t_top:<10.4f} | {t_bot:<10.4f} | {imb:<10.4f}")
        
        done = terminated or truncated

    print("-" * 80)
    print(f"Validation finished. Total Reward: {total_reward:.4f}")
    if step > 0:
        print(f"Final Metrics: Transmission={t_tot:.4f}, Imbalance={imb:.4f}")
        
        print("\nAnalysis:")
        if t_tot > 0.8:
            print("- Transmission is high (>0.8). Good result.")
        else:
            print("- Transmission could be improved.")
            
        if imb < 0.05:
            print("- Output is well-balanced (<0.05).")
        else:
            print("- Output shows some imbalance.")

if __name__ == "__main__":
    validate()
