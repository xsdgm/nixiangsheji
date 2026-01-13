import numpy as np
import autograd.numpy as npa
from autograd import grad
import matplotlib.pyplot as plt
import os

from ceviche import fdfd_hz

from sdf_gen import MMIGeometry

# 配置参数（与 ceviche_sim.py 保持一致）
# 使用无量纲单位 c=1
# 波长 = 1.55 um
OMEGA = 2 * np.pi / 1.55
# DL 来自 MMIGeometry (0.04 um)
# NPML = 20

class MMIAdjointOptimizer:
    def __init__(self, steps=100, learning_rate=0.01):
        self.geo = MMIGeometry()
        self.Nx = self.geo.Nx
        self.Ny = self.geo.Ny
        self.dl = self.geo.dl
        self.npml = 20
        self.steps = steps
        self.lr = learning_rate
        
        # 初始化光源和探测器（静态）
        self._init_ports()
        
        # 设计区域掩码（允许改变的区域）
        # 我们允许优化 MMI 主体区域
        self.design_mask = self._create_design_mask()
        
        # 初始设计变量（归一化到 0 到 1）
        # 在设计区域初始化为 0.5（灰色）以获得最佳梯度流
        # 或者用起始几何形状初始化
        self.design_init = self.geo.get_density() * self.design_mask + 0.5 * self.design_mask * (1 - self.geo.get_density())
        # 实际上，让我们从当前几何形状的种子开始，稍微平滑一下
        self.design_init = self.geo.get_density()
        
    def _init_ports(self):
        # 与 ceviche_sim.py 中的逻辑相同
        nx, ny = self.Nx, self.Ny
        
        # 光源
        self.source_mask = np.zeros((nx, ny), dtype=np.complex128)
        cy = ny // 2
        sigma_pix = 6.0
        y_idx = np.arange(ny)
        # 注意：初始化时使用标准 numpy，而不是 autograd
        gaussian_profile = np.exp(-0.5 * ((y_idx - cy) / sigma_pix)**2)
        src_x = self.npml + 10
        self.source_mask[src_x, :] = gaussian_profile * 10.0
        
        # 探测器
        out_x = nx - self.npml - 5
        mmi_center_y = self.geo.Ly / 2.0
        out_offset = 1.5 
        y_top_idx = int((mmi_center_y + out_offset) / self.dl)
        y_bot_idx = int((mmi_center_y - out_offset) / self.dl)
        
        self.probe_top = np.zeros((nx, ny))
        self.probe_bot = np.zeros((nx, ny))
        self.probe_top[out_x, y_top_idx] = 1.0
        self.probe_bot[out_x, y_bot_idx] = 1.0

    def _create_design_mask(self):
        # 将矩形 MMI 区域定义为设计域
        # 从几何工具获取坐标会更好，但为了与 sdf_gen 保持一致而硬编码
        mask = np.zeros((self.Nx, self.Ny))
        
        # 重新推导 MMI 区域索引
        mmi_length_pix = int(self.geo.mmi_length / self.dl)
        mmi_width_pix = int(self.geo.mmi_width / self.dl)
        
        cx, cy = self.Nx // 2, self.Ny // 2
        
        x_start = cx - mmi_length_pix // 2
        x_end = cx + mmi_length_pix // 2
        y_start = cy - mmi_width_pix // 2
        y_end = cy + mmi_width_pix // 2
        
        mask[x_start:x_end, y_start:y_end] = 1.0
        return mask

    def objective(self, design_variable):
        # design_variable 是 (Nx, Ny) 的密度值数组
        # 可以使用 Sigmoid 投影或简单的截断，但 adam_minimize 处理边界？
        # 不，我们通常将变量映射到密度
        
        # 密度投影（简单的障碍函数是可选的，这里只是使用原始变量解释为密度）
        # 理想情况下我们想要二值化设计，所以稍后可能会添加非二值化惩罚
        # specific_density = npa.clip(design_variable, 0, 1) # Clip 会破坏梯度？使用 sigmoid
        
        density = design_variable # 从简单开始
        
        # epsilon 逻辑
        # eps = mask * (eps_si - eps_bg) + eps_bg
        eps_r = density * (12.0 - 1.0) + 1.0
        
        # 仿真
        # fdfd_hz(omega, dl, eps_r, npml)
        sim = fdfd_hz(OMEGA, self.dl, eps_r, [self.npml, self.npml])
        
        # solve() 返回 (Hz, Ex, Ey) 元组？还是 (Ex, Ey, Hz)？
        # 假设是 ceviche 标准，返回一个元组。
        # 然而，fdfd_hz.solve 通常返回 (Ex, Ey, Hz)。
        # 我们需要计算强度。
        
        out_fields = sim.solve(self.source_mask)
        # 注意：autograd 包装可能会使元组解包变得棘手，如果不小心的话
        # 但是 ceviche 示例通常可以正常解包
        
        # 根据我之前的调试，索引 0 是 Hz。
        Hz = out_fields[0]
        
        # 计算功率（传输）
        # 探测器处的 |Hz|^2
        # 我们必须使用 npa（autograd numpy）进行操作
        
        t_top = npa.sum(npa.abs(Hz * self.probe_top)**2)
        t_bot = npa.sum(npa.abs(Hz * self.probe_bot)**2)
        
        # 优化目标：最大化功率，最小化不平衡
        # Loss = - (Total_T - Penalty * Imbalance^2)
        
        total_t = t_top + t_bot
        imbalance = (t_top - t_bot)**2 # 平方差以严格惩罚
        
        # 权重（任意）
        # 增加不平衡惩罚以强制 50:50
        loss = -1.0 * (10.0 * total_t - 50.0 * imbalance)
        
        return loss

    def callback(self, iteration, of, params):
        print(f"Iter {iteration}: Loss = {of}")
        
        if iteration % 10 == 0:
            # 可视化
            density = params
            plt.figure()
            plt.imshow(density.T, origin='lower', cmap='gray')
            plt.title(f"Iter {iteration}, Loss {of:.4f}")
            plt.colorbar()
            plt.savefig(f"opt_iter_{iteration:03d}.png")
            plt.close()

    def run(self):
        print("Starting Adjoint Optimization...")
        
        # 初始参数
        params = self.design_init
        
        # Adam 参数
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m, v = 0, 0
        
        # 使用 autograd 定义梯度函数
        # 我们需要计算值和梯度
        from autograd import value_and_grad
        
        def obj_wrapper(p):
            # 对参数应用设计掩码
            # 为简单起见，我们将 params 视为完整密度，
            # 但如果我们正确处理，只有设计区域中的梯度才重要。
            # 实际上，让我们将 params 保持为完整网格。
            eff_density = p * self.design_mask + self.geo.get_density() * (1 - self.design_mask)
            return self.objective(eff_density)
        
        value_grad_func = value_and_grad(obj_wrapper)
        
        for i in range(self.steps):
            loss, grads = value_grad_func(params)
            
            # 掩码梯度（仅优化设计区域）
            grads = grads * self.design_mask
            
            # Adam 更新
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * (grads**2)
            m_hat = m / (1 - beta1**(i + 1))
            v_hat = v / (1 - beta2**(i + 1))
            
            # 更新参数
            params = params - self.lr * m_hat / (npa.sqrt(v_hat) + epsilon)
            
            # 投影 / 截断（0 到 1）
            # 对密度很重要
            params = npa.clip(params, 0, 1)
            
            # 回调 / 日志记录
            self.callback(i, loss, params)
            
        return params

if __name__ == "__main__":
    optimizer = MMIAdjointOptimizer(steps=100, learning_rate=0.02)
    final_design = optimizer.run()
    
    # Save Final Result
    np.save("final_design_adjoint.npy", final_design)
    plt.figure()
    plt.imshow(final_design.T, origin='lower', cmap='gray')
    plt.savefig("final_design_adjoint.png")
    print("Optimization Complete.")
