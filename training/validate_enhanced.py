import os
import sys
import numpy as np

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from mmi_env import MMIOptEnv

def validate_enhanced():
    """
    增强版验证脚本：
    - 使用非确定性预测（deterministic=False）增加探索性
    - 配合增大的动作缩放因子（已在mmi_env.py中修改为10.0）
    """
    model_path = "mmi_ppo_model.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {os.path.abspath(model_path)}...")
    
    env = MMIOptEnv()
    
    # 设置平衡的奖励权重（验证模式）
    # Alpha: 传输效率权重
    # Beta:  分光平衡权重（不平衡度惩罚）
    env.alpha = 10.0  # 传输效率
    env.beta = 10.0   # 分光平衡性
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Starting enhanced validation...")
    print("配置:")
    print("  - 非确定性预测: ENABLED (探索模式)")
    print("  - 动作缩放因子: 10.0 (在mmi_env.py中)")
    print(f"  - 奖励权重: Alpha(传输)={env.alpha}, Beta(平衡)={env.beta} (平衡模式)")
    print()
    
    obs, info = env.reset()
    total_reward = 0
    
    MAX_STEPS = 50
    
    print("-" * 100)
    print(f"{'Step':<5} | {'Action Stats':<25} | {'Reward':<12} | {'Total T':<10} | {'T Top':<10} | {'T Bot':<10} | {'Imbalance':<10}")
    print("-" * 100)
    
    initial_density = env.geo.get_density()
    
    for step in range(MAX_STEPS):
        # 使用非确定性预测，增加探索性
        action, _ = model.predict(obs, deterministic=False)
        
        # 统计动作信息
        action_mean = np.mean(action)
        action_std = np.std(action)
        action_min = np.min(action)
        action_max = np.max(action)
        action_stats = f"μ={action_mean:.4f} σ={action_std:.4f}"
        
        # 执行步骤
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        t_tot = info.get('transmission', 0)
        t_top = info.get('t_top', 0)
        t_bot = info.get('t_bot', 0)
        imb = info.get('imbalance', 0)
        
        print(f"{step+1:<5} | {action_stats:<25} | {reward:<12.4f} | {t_tot:<10.4f} | {t_top:<10.4f} | {t_bot:<10.4f} | {imb:<10.4f}")
        
        # 每10步检查一次几何变化
        if (step + 1) % 10 == 0:
            current_density = env.geo.get_density()
            diff = np.sum(np.abs(current_density - initial_density))
            print(f"  --> 累计几何变化: {diff:.0f} 像素 (相对初始状态)")
        
        if terminated or truncated:
            print(f"\nEpisode 在第 {step+1} 步结束")
            break
    
    print("-" * 100)
    print(f"\n验证完成!")
    print(f"  总奖励: {total_reward:.4f}")
    print(f"  最终传输: {t_tot:.4f}")
    print(f"  最终不平衡: {imb:.4f}")
    
    # 最终几何变化统计
    final_density = env.geo.get_density()
    total_change = np.sum(np.abs(final_density - initial_density))
    print(f"  总几何变化: {total_change:.0f} 像素")
    
    print("\n分析:")
    if t_tot > 0.8:
        print("  ✓ 传输性能优秀 (>0.8)")
    else:
        print("  - 传输可以进一步优化")
        
    if imb < 0.05:
        print("  ✓ 输出平衡性好 (<0.05)")
    else:
        print("  - 输出不平衡度较高")
    
    if total_change > 100:
        print(f"  ✓ 几何结构有明显变化 ({total_change:.0f} 像素)")
    else:
        print(f"  ⚠ 几何变化较小 ({total_change:.0f} 像素)")

if __name__ == "__main__":
    validate_enhanced()
