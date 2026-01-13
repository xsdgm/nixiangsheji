import numpy as np
import ceviche
from ceviche import fdfd_hz, fdfd_ez
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class CevicheSim:
    def __init__(self, geometry):
        """
        初始化模拟包装器。
        参数：
            geometry (MMIGeometry): 包含网格信息的MMIGeometry实例。
        """
        self.geo = geometry
        # 使用无量纲单位，其中c=1
        # 长度单位：微米
        self.omega = 2 * np.pi / 1.55 # 1.55微米波长
        self.dl = geometry.dl # 网格尺寸
        self.npml = 40 # PML厚度 (~1.6 um @ dl=0.04, ~0.8um @ dl=0.02)
        
        # 初始化端口和光源位置
        self._init_ports()
        
    def _init_ports(self):
        """定义输入源和输出监测器。"""
        nx, ny = self.geo.Nx, self.geo.Ny
        
        # --- 1. 定义光源 (Source) ---
        self.source_mask = np.zeros((nx, ny), dtype=np.complex128) 
        
        # 默认光源中心在画幅的垂直中心 (Y方向)
        self.cy = ny // 2
        
        # 高斯光源宽度 (半宽)
        # 注意：如果分辨率改变，sigma_pix 也应该相应调整，这里用物理尺寸计算更稳妥
        sigma_um = 0.24  # 对应标准单模波导的模场半径
        sigma_pix = sigma_um / self.dl 
        
        y_idx = np.arange(ny)
        gaussian_profile = np.exp(-0.5 * ((y_idx - self.cy) / sigma_pix)**2)
        
        # 光源 X 位置：PML 之后 10 个像素
        self.src_x = self.npml + 10
        
        # 赋值光源
        # 提高光源强度以获得更好的信噪比
        self.source_mask[self.src_x, :] = gaussian_profile * 1e6 
        
        # --- 2. 定义输出监测器 (Monitors) ---
        # 输出位置：PML 之前 5 个像素
        self.out_x = nx - self.npml - 5
        
        # 输出端口的 Y 坐标偏移量
        mmi_center_idx = ny // 2 
        out_offset_pix = int(self.geo.out_offset / self.geo.dl) 
        self.y_top_idx = mmi_center_idx + out_offset_pix
        self.y_bot_idx = mmi_center_idx - out_offset_pix
        
        # 验证边界
        assert 0 <= self.y_top_idx < ny, f"y_top_idx={self.y_top_idx} 超出范围 [0, {ny})"
        assert 0 <= self.y_bot_idx < ny, f"y_bot_idx={self.y_bot_idx} 超出范围 [0, {ny})"
        
        # 定义监测区域 (用于画图)
        self.global_profile_region = {
            'x_start': self.npml,
            'x_end': nx - self.npml,
            'y_start': self.npml,
            'y_end': ny - self.npml,
            'name': 'Global_Profile'
        }
        
        # 监测线位置
        self.expansion_out_line = self.out_x
        self.throughput_line = self.out_x
        
    def run(self, eps_r, normalization_power=None):
        """
        运行FDFD模拟 (TE模式 / Hz)。
        """
        # --- 安全检查：光源是否在波导芯层？---
        n_at_source = np.sqrt(np.real(eps_r[self.src_x, self.cy]))
        print(f"Debug: Source Location (x={self.src_x}, y={self.cy}), Index n={n_at_source:.3f}")
        
        if n_at_source < 2.0:
            print("\n" + "!"*40)
            print("【严重警告】 光源位于低折射率区域 (n < 2.0)！")
            print("这意味着光源在'空气'或'包层'中。")
            print("请检查 sdf_gen.py 中的几何生成逻辑。")
            print("!"*40 + "\n")

        # --- 运行 FDFD 求解器 (核心修改) ---
        # 使用 fdfd_hz (TE Mode)，标准平面光波导模式
        print(f"Debug: Running FDFD_Hz (TE Mode)... dl={self.dl} um")
        simulation = fdfd_hz(self.omega, self.dl, eps_r, [self.npml, self.npml])
        
        try:
             # fdfd_hz 返回 Hz, Ex, Ey
             Hz, Ex, Ey = simulation.solve(self.source_mask)
        except Exception as e:
             print(f"Error: Simulation failed: {e}")
             return None
        
        # --- 仿真健康检查 ---
        max_field_val = np.max(np.abs(Hz))
        max_field_val = np.max(np.abs(Hz))
        # print(f"Debug: Max Field Strength |Hz| = {max_field_val:.4e}") # Commented out to reduce noise
        
        if max_field_val < 1e-4:
            print("\n" + "X"*50)
            print("【仿真失败】 场强极低 (接近0)！")
            print("原因排查：")
            print("1. ⚠️ 分辨率不够：请去 sdf_gen.py 把 resolution 改为 0.02")
            print("2. 波导断裂：光无法传播。")
            print("X"*50 + "\n")
        
        # --- 数据处理 ---
        
        # 1. Global Field (截取中间区域用于画图)
        region = self.global_profile_region
        global_field = Hz[region['x_start']:region['x_end'], 
                          region['y_start']:region['y_end']]
        
        # 2. Output Analysis (输出截面)
        # 确保使用的是 Hz 分量
        output_field = Hz[self.expansion_out_line, :]
        
        # 积分窗口宽度 (pixels)
        window_r = int(1.0 / self.dl) # 窗口半径 1.0 um
        
        # 计算上端口功率
        top_idx_start = max(0, self.y_top_idx - window_r)
        top_idx_end = min(len(output_field), self.y_top_idx + window_r)
        top_slice = slice(top_idx_start, top_idx_end)
        top_power_abs = np.sum(np.abs(output_field[top_slice])**2) * self.dl
        
        # 计算下端口功率
        bot_idx_start = max(0, self.y_bot_idx - window_r)
        bot_idx_end = min(len(output_field), self.y_bot_idx + window_r)
        bot_slice = slice(bot_idx_start, bot_idx_end)
        bot_power_abs = np.sum(np.abs(output_field[bot_slice])**2) * self.dl
        
        # 计算总通过功率
        total_throughput_abs = np.sum(np.abs(output_field)**2) * self.dl
        
        # --- S 参数计算 (归一化) ---
        if normalization_power is not None and normalization_power > 1e-20:
            S21 = top_power_abs / normalization_power
            S31 = bot_power_abs / normalization_power
            efficiency = total_throughput_abs / normalization_power
            norm_status = "(Normalized)"
        else:
            S21 = top_power_abs 
            S31 = bot_power_abs
            efficiency = total_throughput_abs
            efficiency = total_throughput_abs
            norm_status = "(Absolute - Unnormalized)"

        # 打印关键指标，向用户展示变化
        print(f"  -> Sim Result: S21={S21:.4e}, S31={S31:.4e}, Total={efficiency:.4e}")

        monitors = {
            'global_profile': {
                'field': global_field,
                'power': np.abs(global_field)**2,
                'region': region
            },
            'expansion_out': {
                'field': output_field,
                'S21': S21,
                'S31': S31,
                'position': self.expansion_out_line,
                'status': norm_status
            },
            'throughput': {
                'total_power': total_throughput_abs, 
                'efficiency': efficiency,
                'position': self.throughput_line
            },
            'full_field': Hz,  # 确保返回 Hz
            'source_pos': (self.src_x, self.cy)
        }
        
        return monitors

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    from sdf_gen import MMIGeometry
    
    # 字体设置
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 1. 初始化几何
    # ⚠️⚠️⚠️ 请确保在 sdf_gen.py 里设置了 resolution=0.02 ⚠️⚠️⚠️
    geo = MMIGeometry() 
    
    # 获取器件的介电常数分布
    eps_device = geo.get_permittivity()
    
    # 初始化仿真器
    sim = CevicheSim(geo)
    
    print("运行仿真中...")
    
    # --- 步骤 0: 归一化 (用直波导获取参考功率) ---
    print("\n" + "="*60)
    print("步骤 1/2: 运行直波导归一化...")
    print("="*60)
    eps_straight = geo.get_straight_waveguide()
    norm_res = sim.run(eps_straight, normalization_power=None)
    
    if norm_res is None:
        print("归一化仿真失败！")
        exit(1)
    
    P0 = norm_res['throughput']['total_power']
    print(f"\n归一化功率 P0 = {P0:.6e}\n")

    # --- Debug: 保存直波导场图 ---
    plt.figure(figsize=(10, 4))
    plt.imshow(np.real(norm_res['global_profile']['power']).T, origin='lower', cmap='inferno', aspect='auto', norm=LogNorm(vmin=1e-2))
    plt.title(f'Normalization (Straight WG) |Hz|^2, P0={P0:.2e}')
    plt.colorbar(label='Power')
    plt.savefig('debug_straight_wg.png')
    plt.close()
    print("Debug: 直波导（归一化）场图已保存到 debug_straight_wg.png")
    
    if P0 < 1e-10:
        print("\n" + "!"*60)
        print("【警告】归一化功率过低！可能存在以下问题：")
        print("1. 分辨率仍然太粗 (当前 dl={:.3f} um)".format(sim.dl))
        print("2. 光源与波导未对准")
        print("3. 波导宽度过窄无法支持模式")
        print("!"*60 + "\n")
    
    # --- 步骤 2: 运行器件仿真 ---
    print("\n" + "="*60)
    print("步骤 2/2: 运行 MMI 器件仿真...")
    print("="*60)
    monitors = sim.run(eps_device, normalization_power=P0)
    
    if monitors is None:
        print("仿真失败！")
        exit(1)
    
    # --- 打印结果 ---
    res_out = monitors['expansion_out']
    status = res_out['status']
    print("\n" + "="*60)
    print(f"监测器结果 {status}")
    print("="*60)
    print(f"  S21 (上端口):    {res_out['S21']:.6e}")
    print(f"  S31 (下端口): {res_out['S31']:.6e}")
    print(f"  总计:        {monitors['throughput']['efficiency']:.6e}")
    print("="*60 + "\n")
    
    # --- 可视化 ---
    field = monitors['full_field']
    global_power = monitors['global_profile']['power']
    src_pos = monitors['source_pos']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 图1: 结构 + 光源位置诊断
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.sqrt(np.real(eps_device).T), origin='lower', cmap='gray', aspect='auto')
    ax1.plot(src_pos[0], src_pos[1], 'r*', markersize=15, label='光源位置')
    ax1.set_title(f'几何结构检查 (dl={sim.dl} um)', fontsize=12, fontweight='bold')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='折射率 (n)')
    
    # 图2: MMI 区域光场
    ax2 = fig.add_subplot(gs[0, 1])
    # 使用 log 刻度可以更清晰地看到弱光分布 (开启 LogNorm)
    im2 = ax2.imshow(np.real(global_power).T, origin='lower', cmap='inferno', aspect='auto', norm=LogNorm(vmin=1e-4))
    # im2 = ax2.imshow(np.real(global_power).T, origin='lower', cmap='inferno', aspect='auto')
    ax2.set_title('光场强度 |Hz|^2', fontsize=12)
    plt.colorbar(im2, ax=ax2)
    
    # 图3: 输出截面场分布
    ax3 = fig.add_subplot(gs[1, :])
    out_field = res_out['field']
    y_axis = np.arange(len(out_field)) * sim.dl
    power_profile = np.abs(out_field)**2
    
    ax3.plot(y_axis, power_profile, 'b-', linewidth=2, label='输出功率分布')
    
    # 调试信息
    print(f"\n调试: 输出场形状: {out_field.shape}")
    max_power_idx = np.argmax(power_profile)
    print(f"调试: 最大功率位于 Y={y_axis[max_power_idx]:.3f} um, 值={power_profile[max_power_idx]:.6e}")
    
    ax3.axvline(sim.y_top_idx * sim.dl, color='g', linestyle='--', label='上端口')
    ax3.axvline(sim.y_bot_idx * sim.dl, color='orange', linestyle='--', label='下端口')
    ax3.set_title(f'输出分布 (Hz 模式) {status}', fontsize=12)
    ax3.set_xlabel('Y (微米)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("mmi_simulation_result.png", dpi=150)
    print("结果图像已保存到 mmi_simulation_result.png")
    # plt.show()