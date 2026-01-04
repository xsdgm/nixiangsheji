import gym
import numpy as np
from gym import spaces
from scipy.ndimage import zoom

from sdf_gen import MMIGeometry
from ceviche_sim import CevicheSim

class MMIOptEnv(gym.Env):
    def __init__(self):
        super(MMIOptEnv, self).__init__()
        
        # Initialize Core Components
        self.geo = MMIGeometry()
        self.sim = CevicheSim(self.geo)
        
        # Action Space: 10x10 Control Grid (Deformation)
        # Values between -1 and 1 representing change in SDF
        self.act_res = (10, 10)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.act_res, dtype=np.float32)
        
        # Observation Space: The Density Mask (on the fine grid)
        # Shape matches geometry grid
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.geo.Nx, self.geo.Ny, 1), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 50 # Short episode for shape evolution
        
    def reset(self):
        self.geo.reset_geometry()
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        
        # 1. Process Action (Upscale 10x10 -> MMI Region)
        # We only apply deformation to the central MMI region to avoid breaking ports
        # Extract MMI Region dimensions derived in sdf_gen (need to make public or hardcode)
        # For prototype: simple global upscale and mask out ports
        
        action_map = action # (10, 10)
        
        # Calculate zoom factors to match full grid
        zoom_x = self.geo.Nx / self.act_res[0]
        zoom_y = self.geo.Ny / self.act_res[1]
        
        # Upscale action to full grid size
        delta_sdf = zoom(action_map, (zoom_x, zoom_y), order=1)
        
        # Ensure sizes match exactly (rounding errors)
        delta_sdf = delta_sdf[:self.geo.Nx, :self.geo.Ny]
        # Pad if needed
        if delta_sdf.shape != (self.geo.Nx, self.geo.Ny):
            pad_x = self.geo.Nx - delta_sdf.shape[0]
            pad_y = self.geo.Ny - delta_sdf.shape[1]
            delta_sdf = np.pad(delta_sdf, ((0,pad_x),(0,pad_y)), 'constant')

        # Apply spatial mask to delta_sdf so we don't move input/output ports
        # (Simple box mask for center region)
        center_mask = np.zeros_like(delta_sdf)
        mmi_start = int(self.geo.Nx * 0.2)
        mmi_end = int(self.geo.Nx * 0.8)
        center_mask[mmi_start:mmi_end, :] = 1.0
        
        final_delta = delta_sdf * center_mask * 0.5 # Scale deformation strength
        
        # 2. Update Geometry
        self.geo.update_sdf(final_delta)
        
        # 3. Simulation
        eps = self.geo.get_permittivity()
        (t_top, t_bot), fields = self.sim.run(eps)
        
        # 4. Reward
        # Goal: Maximize total transmission AND Minimize imbalance
        # Reward = alpha * Total_T - beta * |T_top - T_bot|
        total_t = t_top + t_bot
        imbalance = abs(t_top - t_bot)
        
        alpha = 10.0
        beta = 20.0
        
        reward = alpha * total_t - beta * imbalance
        
        # 5. Done Condition
        done = self.current_step >= self.max_steps
        
        info = {
            'transmission': total_t,
            't_top': t_top,
            't_bot': t_bot,
            'imbalance': imbalance
        }
        
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Return density as (Nx, Ny, 1) image
        obs = self.geo.get_density()
        return np.expand_dims(obs, axis=-1)

    def render(self, mode='human'):
        # Optional: Visualize current state
        pass
