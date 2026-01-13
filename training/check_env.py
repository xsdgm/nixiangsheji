from stable_baselines3.common.env_checker import check_env
from mmi_env import MMIOptEnv

def main():
    print("Checking MMI Environment...")
    env = MMIOptEnv()
    
    # Check compliance with Gymnasium API
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Manual stepped check
    print("\nManual Step Check:")
    obs, info = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Reward: {reward}")
    print(f"Step Info: {info}")
    print("Verification Complete.")

if __name__ == "__main__":
    main()
