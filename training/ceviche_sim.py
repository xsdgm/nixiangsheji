import numpy as np
import ceviche
from ceviche import fdfd_hz, fdfd_ez
import matplotlib.pyplot as plt

class CevicheSim:
    def __init__(self, geometry):
        """
        初始化模拟包装器。
        参数：
            geometry (MMIGeometry): 包含网格信息的MMIGeometry实例。
        """
        self.geo = geometry
        self.geo = geometry
        # 使用无量纲单位，其中c=1
        # 长度单位：微米
        self.omega = 2 * np.pi / 1.55 # 1.55微米波长，c=1
        self.dl = geometry.dl # 已转换为微米
        self.npml = 20 # 完美匹配层（PML）厚度（像素）
        
        # 源和探针定义
        self._init_ports()
        
    def _init_ports(self):
        """定义输入源和输出监测器。"""
        nx, ny = self.geo.Nx, self.geo.Ny
        
        # 输入源（左侧）
        # 使用高斯分布（复数）- 恢复到运行185的配置
        self.source_mask = np.zeros((nx, ny), dtype=np.complex128) 
        cy = ny // 2
        
        sigma_pix = 6.0
        y_idx = np.arange(ny)
        gaussian_profile = np.exp(-0.5 * ((y_idx - cy) / sigma_pix)**2)
        
        src_x = self.npml + 10
        self.source_mask[src_x, :] = gaussian_profile * 10.0
        
        # 输出监测器（右侧）
        out_x = nx - self.npml - 5
        mmi_center_y = self.geo.Ly / 2.0
        out_offset = 1.5 
        y_top_idx = int((mmi_center_y + out_offset) / self.geo.dl)
        y_bot_idx = int((mmi_center_y - out_offset) / self.geo.dl)
        
        self.probe_top = np.zeros((nx, ny))
        self.probe_bot = np.zeros((nx, ny))
        self.probe_top[out_x, y_top_idx] = 1.0
        self.probe_bot[out_x, y_bot_idx] = 1.0
        
    def run(self, eps_r):
        """
        运行FDFD模拟。
        参数：
             eps_r (np.array): 相对介电常数分布 (Nx, Ny)。
        返回值：
             transmission (float): 输出端口的总传输。
             fields (np.array): 场分布 (Hz)。
        """
        # 运行FDFD（TE的Hz极化）
        simulation = fdfd_hz(self.omega, self.dl, eps_r, [self.npml, self.npml])
        
        try:
             # fdfd_hz返回 (Ex, Ey, Hz) 元组
             fields_tuple = simulation.solve(self.source_mask)
             Hz = fields_tuple[0] # 此版本中第一个分量包含Hz场
        except Exception as e:
             # 稳定性回退
             print(f"警告：模拟失败：{e}")
             return 0.0, np.zeros_like(self.source_mask)
        
        # 计算探针处的功率/强度
        t_top = np.sum(np.abs(Hz * self.probe_top)**2)
        t_bot = np.sum(np.abs(Hz * self.probe_bot)**2)
        
        # 返回各自的传输以用于奖励计算
        return (t_top, t_bot), Hz

if __name__ == "__main__":
    from sdf_gen import MMIGeometry
    geo = MMIGeometry()
    sim = CevicheSim(geo)
    eps = geo.get_permittivity()
    
    T, field = sim.run(eps)
    print(f"基准传输：{T}")
    
    plt.imshow(np.real(field).T, origin='lower', cmap='RdBu')
    plt.savefig("sim_test.png")
