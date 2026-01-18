import numpy as np
import matplotlib.pyplot as plt

class MMIGeometry:
    def __init__(self, region_size=(50.0, 10.0), resolution=0.04):
        """
        初始化 MMI 几何生成器。
        
        参数：
            region_size (tuple): 物理尺寸 (Lx, Ly)，单位微米。
            resolution (float): 网格步长，单位微米。
        """
        self.Lx, self.Ly = region_size
        self.dl = resolution
        self.Nx = int(self.Lx / self.dl)
        self.Ny = int(self.Ly / self.dl)
        
        # 创建网格
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # MMI 设计参数（优化后 - 逆向设计初始值）
        # 增大区域以提供更多自由度
        self.wg_width = 0.5
        self.mmi_width = 6.0   # 宽度增加到 6.0um
        self.mmi_length = 35.0 # 长度增加到 35.0um
        self.taper_width = 1.0
        self.taper_length = 2.0
        self.out_offset = 1.5 # 调整输出偏移以适应更宽的 MMI
        
        # 初始化基础 SDF
        self.sdf = np.zeros((self.Nx, self.Ny))
        self.reset_geometry()

    def reset_geometry(self):
        """重置 SDF，生成初始 MMI 形状。"""
        # 清空场景
        self.sdf = np.full((self.Nx, self.Ny), -1.0)
        
        # 中心坐标
        cy = self.Ly / 2.0
        cx = self.Lx / 2.0
        
        # 1. 输入波导（左侧）
        # 范围：x=0 到 MMI 起点
        input_y_dist = np.abs(self.Y - cy) - (self.wg_width / 2.0)
        
        # 2. MMI 主体（中心）
        mmi_start_x = (self.Lx - self.mmi_length) / 2.0
        mmi_end_x = mmi_start_x + self.mmi_length
        mmi_y_dist = np.abs(self.Y - cy) - (self.mmi_width / 2.0)
        
        # 3. 形状组合（并集）
        # 采用简单矩形近似：SDF>0 表示硅，SDF<0 表示包层（空气/SiO2）
        
        # 输入波导区域
        mask_input = (self.X < mmi_start_x) & (input_y_dist < 0)
        
        # MMI 主体区域
        mask_mmi = (self.X >= mmi_start_x) & (self.X <= mmi_end_x) & (mmi_y_dist < 0)
        
        # 输出波导（上下两臂）
        mask_out_top = (self.X > mmi_end_x) & (np.abs(self.Y - (cy + self.out_offset)) < self.wg_width/2.0)
        mask_out_bot = (self.X > mmi_end_x) & (np.abs(self.Y - (cy - self.out_offset)) < self.wg_width/2.0)
        
        self.mask = mask_input | mask_mmi | mask_out_top | mask_out_bot
        
        # 将二值掩码转换成粗略 SDF（1 为器件区，-1 为包层）
        self.sdf = np.where(self.mask, 1.0, -1.0)

    def update_sdf(self, action_map):
        """
        根据 action_map 对 SDF 做增量更新。
        为简单起见，这里把 action_map 视作作用在 MMI 区域的加性扰动。
        """
        # 假设 action_map 已与 MMI 区域尺寸匹配（或已插值）
        # 使用更大的缩放因子，使动作能够实际改变几何
        self.sdf += action_map * 1.0  # 增大缩放因子以产生可见的几何变化

    def get_density(self):
        """基于 SDF>0 返回二值密度（0 或 1）。"""
        return (self.sdf > 0).astype(float)
        
    def get_permittivity(self, eps_struct=12.0, eps_bg=1.0):
        """返回相对介电常数分布。"""
        mask = self.get_density()
        return mask * (eps_struct - eps_bg) + eps_bg
    
    def get_straight_waveguide(self, eps_struct=12.0, eps_bg=1.0):
        """生成直波导用于归一化。"""
        cy = self.Ly / 2.0
        wg_y_dist = np.abs(self.Y - cy) - (self.wg_width / 2.0)
        mask_straight = wg_y_dist < 0
        return np.where(mask_straight, eps_struct, eps_bg)

if __name__ == "__main__":
    geo = MMIGeometry()
    plt.imshow(geo.get_density().T, origin='lower')
    plt.savefig("mmi_init.png")
    print("几何测试完成。")
