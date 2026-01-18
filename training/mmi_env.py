import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import zoom

from sdf_gen import MMIGeometry
from ceviche_sim import CevicheSim

class MMIOptEnv(gym.Env):
    def __init__(self):
        super(MMIOptEnv, self).__init__()
        
        # 初始化核心组件
        self.geo = MMIGeometry()
        self.sim = CevicheSim(self.geo)
        
        # 动作空间：10x10 控制网格（形变）
        # 值在 -1 和 1 之间，表示 SDF 的变化
        self.act_res = (10, 10)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.act_res, dtype=np.float32)
        
        # 观察空间：密度掩码（在细网格上）
        # 形状与几何网格匹配
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.geo.Nx, self.geo.Ny, 1), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 50 # 形状演化的短回合
        
        # 动态奖励权重 (将在训练中调整)
        # 初始阶段：极大偏重通光率，忽略不平衡
        self.alpha = 20.0 # 通光率权重
        self.beta = 0.0   # 不平衡权重 (初始为0)
        
        # 记录历史最佳结构 (Elitism)
        self.best_reward = -float('inf')
        self.best_sdf = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 混合重置策略 (Hybrid Reset Strategy)
        rand_val = np.random.random()
        
        reset_type = ""
        
        if rand_val < 0.2:
            # 20% 概率：完全重置 (Hard Reset) - 回到初始矩形
            # 给予全新的探索机会
            self.geo.reset_geometry()
            reset_type = "Hard Reset (初始形状)"
            
        elif rand_val < 0.5 and self.best_sdf is not None:
            # 30% 概率 (0.2~0.5)：回滚到最佳 (Rollback) 
            # 从已知最好的点重新出发
            self.geo.sdf = self.best_sdf.copy()
            reset_type = "Rollback (回滚到最佳)"
            
        else:
            # 50% 概率 (0.5~1.0)：软重置 (Soft Reset)
            # 保留当前几何，继续微调 (Continuous Optimization)
            # 注意：如果没有 best_sdf (刚开始训练)，也会落入这里，相当于继续优化
            reset_type = "Soft Reset (继续优化)"
            # 不做任何改变，保留 self.geo.sdf
            
        print(f"Episode Reset: {reset_type} | Step {self.current_step} -> 0")
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. 处理动作（将 10x10 上采样到 MMI 区域）
        # 我们只对中心的 MMI 区域应用形变，以避免破坏端口
        # 提取 MMI 区域尺寸在 sdf_gen 中推导（需要公开或硬编码）
        # 原型：简单的全局上采样并掩码掉端口
        
        action_map = action # (10, 10)
        
        # 计算缩放因子以匹配完整网格
        zoom_x = self.geo.Nx / self.act_res[0]
        zoom_y = self.geo.Ny / self.act_res[1]
        
        # 将动作上采样到完整网格大小
        delta_sdf = zoom(action_map, (zoom_x, zoom_y), order=1)
        
        # 确保尺寸完全匹配（舍入误差）
        delta_sdf = delta_sdf[:self.geo.Nx, :self.geo.Ny]
        # 如果需要则填充
        if delta_sdf.shape != (self.geo.Nx, self.geo.Ny):
            pad_x = self.geo.Nx - delta_sdf.shape[0]
            pad_y = self.geo.Ny - delta_sdf.shape[1]
            delta_sdf = np.pad(delta_sdf, ((0,pad_x),(0,pad_y)), 'constant')

        # 对 delta_sdf 应用空间掩码，这样我们就不会移动输入/输出端口
        # （中心区域的简单盒式掩码）
        center_mask = np.zeros_like(delta_sdf)
        mmi_start = int(self.geo.Nx * 0.2)
        mmi_end = int(self.geo.Nx * 0.8)
        center_mask[mmi_start:mmi_end, :] = 1.0
        
        final_delta = delta_sdf * center_mask * 10.0 # 缩放形变强度（增大以产生可见变化）
        
        # 2. 更新几何
        self.geo.update_sdf(final_delta)
        
        # 3. 仿真
        eps = self.geo.get_permittivity()
        simulation_results = self.sim.run(eps)
        
        # 从字典中提取结果
        # sim.run 返回 monitors 字典
        monitor_out = simulation_results['expansion_out']
        t_top = monitor_out['S21']
        t_bot = monitor_out['S31']
        
        # 如果需要 fields，可以从 simulation_results['full_field'] 获取，但目前不需要
        
        # 4. 奖励
        # 目标：最大化总传输并最小化不平衡
        # Reward = alpha * Total_T - beta * |T_top - T_bot|
        # 注意：alpha 和 beta 现在是类属性，由外部 Callback 动态调整
        
        total_t = t_top + t_bot
        imbalance = abs(t_top - t_bot)
        
        reward = self.alpha * total_t - self.beta * imbalance
        
        # --- 更新最佳记录 ---
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_sdf = self.geo.sdf.copy()
            print(f"  >>> New Best Found! Reward: {reward:.4f} (Trans={total_t:.4f}, Imbal={imbalance:.4f})")
        # --------------------
        
        # 5. 完成条件
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'transmission': total_t,
            't_top': t_top,
            't_bot': t_bot,
            'imbalance': imbalance
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # 返回密度为 (Nx, Ny, 1) 图像
        obs = self.geo.get_density()
        return np.expand_dims(obs, axis=-1).astype(np.float32)

    def render(self, mode='human'):
        # 可选：可视化当前状态
        pass
