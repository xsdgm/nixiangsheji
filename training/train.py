import numpy as np
import os
import matplotlib.pyplot as plt

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable Baselines3 not found. Running in Random Mode.")

from mmi_env import MMIOptEnv

def main():
    # 1. Initialize Environment
    env = MMIOptEnv()
    
    # 2. Check Environment Compliance
    if SB3_AVAILABLE:
        print("Checking Gym Environment compliance...")
        try:
            check_env(env)
            print("Environment is compliant.")
        except Exception as e:
            print(f"Warning: Environment check warning: {e}")

    # 3. Create Log Directory
    log_dir = "./logs"
    models_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if SB3_AVAILABLE:
        # 4. Train PPO Agent
        print("Starting PPO Training...")
        # Save a checkpoint every 500 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=500,
            save_path=models_dir,
            name_prefix="ppo_mmi"
        )
        
        # Train for 10000 steps (Formal Training)
        # n_steps=20 is kept small for frequent updates given slow sim
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=20, batch_size=20)
        
        model.learn(total_timesteps=10000, callback=checkpoint_callback)
        
        print("Training finished. Saving final model...")
        model.save("ppo_mmi_splitter_final")
        
        # 5. Evaluate
        obs = env.reset()
        print("Evaluating trained agent...")
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: Reward={reward:.4f}, Transmission={info['transmission']:.4f}")
            if done:
                obs = env.reset()
    else:
        # Random Agent Loop
        print("Running Random Agent Loop...")
        obs = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: Reward={reward:.4f}, Transmission={info['transmission']:.4f}")
            if done:
                obs = env.reset()

    # Visualize final state
    plt.figure()
    plt.title("Final MMI Design")
    plt.imshow(env.geo.get_density().T, origin='lower', cmap='binary')
    plt.colorbar(label='Density')
    plt.savefig("final_design.png")
    print("Saved final_design.png")

if __name__ == "__main__":
    main()
