"""
完整的Y分支逆向设计示例 - 使用Lumerical的lumopt库
本脚本展示如何使用Ansys Lumerical的正式优化框架
"""

import numpy as np
import sys
import os
os.environ['LUMERICAL_GPU'] = '1'

# 设置Lumerical API路径
lumerical_path = r"D:\Lumerical\v202\api\python"
if lumerical_path not in sys.path:
    sys.path.append(lumerical_path)


try:
    # 尝试导入lumopt
    from lumopt.geometries.polygon import FunctionDefinedPolygon
    from lumopt.utilities.load_lumerical_scripts import load_from_lsf
    from lumopt.figures_of_merit.modematch import ModeMatch
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers
    from lumopt.optimization import Optimization
    from lumopt.utilities.wavelengths import Wavelengths
    from lumopt.utilities.materials import Material
    
    LUMOPT_AVAILABLE = True
    print("成功导入lumopt库")
    
except ImportError as e:
    LUMOPT_AVAILABLE = False
    print("警告: 未找到lumopt库")
    print(f"错误信息: {e}")




def splitter_function(params):
    """
    定义Y分支的可优化几何体 - Y型分支设计
    
    参数:
        params: 优化变量数组
                前一半控制外边缘 (Outer Top)
                后一半控制内边缘 (Inner Split)
        
    返回:
        polygon_points: 多边形顶点数组
    """
    import scipy.interpolate as sp
    
    # 确定参数数量分配
    n_total = len(params)
    n_outer = n_total // 2
    n_inner = n_total - n_outer
    
    params_outer = params[:n_outer]
    params_inner = params[n_outer:]
    
    # ==========================================
    # 1. 构建上边缘 (Outer Top)
    # ==========================================
    # x范围: -1.0 到 1.0
    outer_x_knots = np.linspace(-1.0e-6, 1.0e-6, n_outer)
    
    # 扩展点以确保连接
    points_x_outer = np.concatenate((
        [outer_x_knots.min() - 0.1e-6],  # 输入侧延伸
        outer_x_knots,
        [outer_x_knots.max() + 0.1e-6]   # 输出侧延伸
    ))
    
    points_y_outer = np.concatenate((
        [params_outer[0]],      # 左端保持输入波导宽度
        params_outer,           # 中间优化区域
        [params_outer[-1]]      # 右端保持输出波导位置
    ))
    
    # 样条插值 - 外边缘
    n_interp = 100
    poly_x_outer = np.linspace(min(points_x_outer), max(points_x_outer), n_interp)
    interp_outer = sp.interpolate.interp1d(points_x_outer, points_y_outer, kind='cubic')
    poly_y_outer = interp_outer(poly_x_outer)
    
    # ==========================================
    # 2. 构建内边缘 (Inner Split)
    # ==========================================
    # x范围: 0.0 到 1.0 (仅在分叉后存在)
    inner_x_knots = np.linspace(0.0e-6, 1.0e-6, n_inner)
    
    # 扩展点
    points_x_inner = np.concatenate((
        [inner_x_knots.min() - 0.05e-6], # 向左微延，确保尖端闭合
        inner_x_knots,
        [inner_x_knots.max() + 0.1e-6]   # 输出侧延伸
    ))
    
    points_y_inner = np.concatenate((
        [params_inner[0]],      # 分叉点
        params_inner,           # 中间优化区域
        [params_inner[-1]]      # 右端内侧位置
    ))
    
    # 样条插值 - 内边缘
    # 注意：内边缘的插值范围只在 x >= 0
    poly_x_inner = np.linspace(0.0, max(points_x_inner), 50)
    interp_inner = sp.interpolate.interp1d(points_x_inner, points_y_inner, kind='cubic')
    poly_y_inner = interp_inner(poly_x_inner)
    
    # ==========================================
    # 3. 组合多边形
    # ==========================================
    
    # 上半部分外边缘点 (从左到右)
    # 过滤掉 x > 0 的部分用于和内边缘拼接? 不需要，直接用全长
    upper_outer = list(zip(poly_x_outer, poly_y_outer))
    
    # 上半部分内边缘点 (从右到左)
    upper_inner = list(zip(poly_x_inner, poly_y_inner))[::-1]
    
    # 下半部分内边缘点 (从左到右，对称)
    lower_inner = [(x, -y) for x, y in zip(poly_x_inner, poly_y_inner)]
    
    # 下半部分外边缘点 (从右到左，对称)
    lower_outer = [(x, -y) for x, y in upper_outer[::-1]]
    
    # 组合所有点
    final_points = []
    final_points.extend(upper_outer)
    final_points.extend(upper_inner)
    final_points.extend(lower_inner)
    final_points.extend(lower_outer)
    
    return np.array(final_points)


def run_lumopt_optimization():
    """使用lumopt运行完整的优化"""

    if not LUMOPT_AVAILABLE:
        print("\n错误: lumopt库不可用，无法运行优化")
        print("请使用 y_branch_optimization.py 中的简化版本")
        return None
    
    print("\n" + "="*60)
    print("使用Lumerical lumopt进行Y分支优化")
    print("="*60)

    # 1. 定义基础仿真（可以从lsf脚本加载）
    print("\n步骤1: 设置基础仿真")
    
    # 这里可以加载预定义的仿真文件
    # script = load_from_lsf('y_branch_base.lsf')
    
    # 或者使用函数定义
    from y_branch_base_setup import setup_base_simulation
    
    def setup_sim(sim):
        setup_base_simulation(sim, 1300e-9, 1800e-9)
    
    # 2. 定义优化几何体
    print("\n步骤2: 定义优化几何体")
    
    # 初始参数：Y型分支设计
    # 我们现在同时优化外边缘和内边缘
    n_points_outer = 10
    n_points_inner = 10
    n_points = n_points_outer + n_points_inner
    
    # 初始猜测
    # 外边缘: 0.25 -> 1.25 (x: -1.0 -> 1.0)
    initial_outer = np.linspace(0.25e-6, 1.25e-6, n_points_outer)
    # 内边缘: 0.0 -> 0.75 (x: 0.0 -> 1.0)
    initial_inner = np.linspace(0.0, 0.75e-6, n_points_inner)
    
    initial_points_y = np.concatenate([initial_outer, initial_inner])

    # 参数边界
    # 外边缘边界
    bounds_outer = [(0.2e-6, 1.5e-6)] * n_points_outer
    bounds_outer[0] = (0.24e-6, 0.26e-6)   # 左端固定 (输入)
    bounds_outer[-1] = (1.2e-6, 1.3e-6)    # 右端固定 (输出外侧)
    
    # 内边缘边界
    # 允许一定的自由度，但必须保持在合理范围内
    bounds_inner = [(-0.1e-6, 0.8e-6)] * n_points_inner
    bounds_inner[0] = (-0.05e-6, 0.05e-6)  # 分叉点固定在0附近
    bounds_inner[-1] = (0.7e-6, 0.8e-6)    # 右端固定 (输出内侧)
    
    bounds = bounds_outer + bounds_inner
    
    print(f"  优化参数数量: {n_points} (外边缘 {n_points_outer} + 内边缘 {n_points_inner})")
    print(f"  初始外边缘: {initial_outer[0]*1e6:.3f}μm -> {initial_outer[-1]*1e6:.3f}μm")
    print(f"  初始内边缘: {initial_inner[0]*1e6:.3f}μm -> {initial_inner[-1]*1e6:.3f}μm")
    
    # 材料定义
    # 使用FDTD自带材料库名称
    eps_in = Material(name='Si (Silicon) - Palik', mesh_order=2)
    eps_out = Material(name='SiO2 (Glass) - Palik', mesh_order=3)
    
    # 创建多边形几何体
    depth = 220.0e-9  # 220nm厚度
    
    polygon = FunctionDefinedPolygon(
        func=splitter_function,
        initial_params=initial_points_y,
        bounds=bounds,
        z=0.0,
        depth=depth,
        eps_out=eps_out,
        eps_in=eps_in,
        edge_precision=5,
        dx=1.0e-9
    )
    
    # 3. 定义品质因数(FOM)
    print("\n步骤3: 定义品质因数")
    
    # 波长范围
    wavelengths = Wavelengths(start=1300e-9, stop=1800e-9, points=21)
    
    # FOM: 模式匹配 - 最大化传输到上路输出端口
    # 目标：最大化光功率（目标传输率为1）
    # 由于对称性，上路最大化意味着下路也会最大化
    fom = ModeMatch(
        monitor_name='fom_monitor_1',
        mode_number='fundamental mode',
        direction='Forward',
        target_T_fwd=lambda wl: np.ones(wl.size),  # 目标传输率为1
        norm_p=1
    )
    
    # 4. 定义优化器
    print("\n步骤4: 设置优化器")
    
    optimizer = ScipyOptimizers(
        max_iter=30,  # 增加迭代次数
        method='L-BFGS-B',
        scaling_factor=1.0,
        pgtol=1e-20,  # 更严格的收敛条件
        ftol=1e-9    # 函数值收敛容限
    )
    
    # 5. 创建优化对象
    print("\n步骤5: 创建优化对象")
    
    opt = Optimization(
        base_script=setup_sim,
        wavelengths=wavelengths,
        fom=fom,
        geometry=polygon,
        optimizer=optimizer,
        use_var_fdtd=False,  # 改为3D FDTD优化
        hide_fdtd_cad=False,
        use_deps=True,
        plot_history=True,  # 启用历史绘图
    )

    # 6. 运行优化
    print("\n步骤6: 开始优化...")
    print("="*60)
    
    try:
        opt.run()
        print("\n优化完成！")
    except Exception as e:
        print(f"\n优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    return opt





if __name__ == "__main__":
    """主程序入口"""
    
    print("\n" + "="*60)
    print("Y分支逆向设计 - Lumerical lumopt版本")
    print("="*60)
    
    if LUMOPT_AVAILABLE:
        print("\n检测到lumopt库，直接启动完整优化")
        try:
            run_lumopt_optimization()
            print("\n优化结果已保存")
        except Exception as e:
            print(f"\n优化过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n未检测到lumopt库，无法运行优化")
    
    print("\n程序执行完成！")
