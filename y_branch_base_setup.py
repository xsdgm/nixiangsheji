"""
Y分支基础仿真设置
同时支持MODE（2.5D）和FDTD（3D）
"""

import numpy as np
import sys
import os

# 设置Lumerical API路径
lumerical_path = r"D:\Lumerical\v202\api\python"
if lumerical_path not in sys.path:
    sys.path.append(lumerical_path)

import lumapi


def setup_base_simulation_mode(sim, wavelength_start=1300e-9, wavelength_stop=1800e-9):
    """
    为MODE设置Y分支基础仿真环境 (2.5D)
    
    参数:
        sim: MODE仿真对象
        wavelength_start: 起始波长 (m)
        wavelength_stop: 终止波长 (m)
    """

    # 帮助定位属性设置错误，便于排查接口字段名
    def safe_set(name, value):
        try:
            sim.set(name, value)
        except Exception as e:
            print(f"[safe_set] set('{name}', {value}) 失败: {e}")
            raise

    # 可选字段，失败时仅提示不终止
    def soft_set(name, value):
        try:
            sim.set(name, value)
        except Exception as e:
            print(f"[soft_set] 跳过 set('{name}', {value}): {e}")
    
    # 仿真区域参数
    sim_length = 6e-6  # 6微米
    sim_width = 6e-6   # 6微米
    
    # 波导参数
    waveguide_width = 0.5e-6   # 500nm
    waveguide_height = 0.22e-6  # 220nm (SOI标准)
    waveguide_spacing = 2e-6    # 输出波导间距
    
    # 材料设置
    n_si = 3.48   # 硅
    n_sio2 = 1.44  # 二氧化硅
    
    print("MODE 2.5D 仿真设置")
    print("="*50)
    print("设置仿真区域...")
    
    # 添加varFDTD区域（MODE中用于参数优化）
    sim.addvarfdtd()
    safe_set("x", 0)
    safe_set("y", 0)
    safe_set("x span", sim_length)
    safe_set("y span", sim_width)
    safe_set("mesh accuracy", 3)
    safe_set("simulation time", 1000e-15)  # 1ps
    
    # 设置边界条件
    safe_set("x min bc", "PML")
    safe_set("x max bc", "PML")
    safe_set("y min bc", "PML")
    safe_set("y max bc", "PML")
    
    print("添加材料...")
    
    # 添加基底材料 (SiO2) - 背景
    sim.addrect()
    safe_set("name", "substrate")
    safe_set("x", 0)
    safe_set("y", 0)
    safe_set("x span", sim_length)
    safe_set("y span", sim_width)
    safe_set("index", n_sio2)
    safe_set("alpha", 0.3)  # 透明度
    
    print("添加输入波导...")
    
    # 输入波导
    sim.addrect()
    safe_set("name", "input_waveguide")
    safe_set("x", -sim_length/2 + 1e-6)
    safe_set("x span", 2e-6)
    safe_set("y", 0)
    safe_set("y span", waveguide_width)
    safe_set("index", n_si)
    
    print("添加输出波导...")
    
    # 输出波导1 (上方) - 与输入波导对齐后逐渐分开
    sim.addrect()
    safe_set("name", "output_waveguide_1")
    safe_set("x", sim_length/2 - 1e-6)
    safe_set("x span", 2e-6)
    safe_set("y", waveguide_spacing/2)  # 最终位置
    safe_set("y span", waveguide_width)
    safe_set("index", n_si)
    
    # 输出波导2 (下方) - 与输入波导对齐后逐渐分开
    sim.addrect()
    safe_set("name", "output_waveguide_2")
    safe_set("x", sim_length/2 - 1e-6)
    safe_set("x span", 2e-6)
    safe_set("y", -waveguide_spacing/2)  # 最终位置
    safe_set("y span", waveguide_width)
    safe_set("index", n_si)
    
    print("添加光源...")
    
    # 添加模式光源（TE基模）
    sim.addmode()
    soft_set("injection axis", "x-axis")
    soft_set("direction", "Forward")
    safe_set("x", -sim_length/2 + 0.5e-6)
    safe_set("y", 0)
    safe_set("y span", 2e-6)
    soft_set("wavelength start", wavelength_start)
    soft_set("wavelength stop", wavelength_stop)
    
    print("添加监视器...")
    
    # 添加优化区域字段监视器
    sim.addpower()
    safe_set("name", "opt_fields")
    safe_set("x", 0)
    safe_set("x span", 2.5e-6)
    safe_set("y", 0)
    safe_set("y span", 3e-6)
    
    # 添加FOM监视器 - 输出1
    sim.addpower()
    safe_set("name", "fom_monitor_1")
    safe_set("x", sim_length/2 - 0.5e-6)
    safe_set("y", waveguide_spacing/2)
    safe_set("y span", 1.5e-6)
    
    # 添加FOM监视器 - 输出2
    sim.addpower()
    safe_set("name", "fom_monitor_2")
    safe_set("x", sim_length/2 - 0.5e-6)
    safe_set("y", -waveguide_spacing/2)
    safe_set("y span", 1.5e-6)
    
    # 添加网格覆盖
    sim.addmesh()
    safe_set("name", "opt_mesh")
    safe_set("x", 0)
    safe_set("x span", 2.5e-6)
    safe_set("y", 0)
    safe_set("y span", 3e-6)
    safe_set("dx", 20e-9)  # 20nm网格
    safe_set("dy", 20e-9)
    
    print("MODE基础仿真设置完成！")
    print("="*50)
    
    return {
        'sim_length': sim_length,
        'sim_width': sim_width,
        'waveguide_width': waveguide_width,
        'waveguide_height': waveguide_height,
        'waveguide_spacing': waveguide_spacing,
        'n_si': n_si,
        'n_sio2': n_sio2
    }


def setup_base_simulation_fdtd(sim, wavelength_start=1300e-9, wavelength_stop=1800e-9):
    """
    为FDTD设置Y分支基础仿真环境 (3D)
    
    参数:
        sim: FDTD仿真对象
        wavelength_start: 起始波长 (m)
        wavelength_stop: 终止波长 (m)
    """
    
    # 仿真区域参数
    sim_length = 6e-6  # 6微米
    sim_width = 6e-6   # 6微米
    sim_height = 3e-6  # 3微米
    
    # 波导参数
    waveguide_width = 0.5e-6   # 500nm
    waveguide_height = 0.22e-6  # 220nm
    waveguide_spacing = 2e-6    # 输出波导间距
    
    # 材料设置
    n_si = 3.48
    n_sio2 = 1.44
    
    print("FDTD 3D 仿真设置")
    print("="*50)
    print("设置仿真区域...")
    
    # 添加FDTD区域
    sim.addfdtd()
    sim.set("x", 0)
    sim.set("y", 0)
    sim.set("z", 0)
    sim.set("x span", sim_length)
    sim.set("y span", sim_width)
    sim.set("z span", sim_height)
    sim.set("mesh accuracy", 3)
    sim.set("simulation time", 1000e-15)  # 1ps
    

    # 设置边界条件
    sim.set("x min bc", "PML")
    sim.set("x max bc", "PML")
    sim.set("y min bc", "PML")
    sim.set("y max bc", "PML")
    sim.set("z min bc", "PML")
    sim.set("z max bc", "PML")
    
    print("添加材料...")
    
    # 添加基底
    sim.addrect()
    sim.set("name", "substrate")
    sim.set("x", 0)
    sim.set("y", 0)
    sim.set("z", 0)
    sim.set("x span", sim_length)
    sim.set("y span", sim_width)
    sim.set("z span", sim_height)
    sim.set("index", n_sio2)
    sim.set("alpha", 0.3)
    
    print("添加输入波导...")
    
    # 输入波导
    sim.addrect()
    sim.set("name", "input_waveguide")
    sim.set("x", -sim_length/2 + 1e-6)
    sim.set("x span", 2e-6)
    sim.set("y", 0)
    sim.set("y span", waveguide_width)
    sim.set("z", 0)
    sim.set("z span", waveguide_height)
    sim.set("index", n_si)
    
    print("添加输出波导...")
    
    # 输出波导1 (上方) - 与输入波导对齐后逐渐分开
    sim.addrect()
    sim.set("name", "output_waveguide_1")
    sim.set("x", sim_length/2 - 1e-6)
    sim.set("x span", 2e-6)
    sim.set("y", waveguide_spacing/2)  # 最终位置
    sim.set("y span", waveguide_width)
    sim.set("z", 0)
    sim.set("z span", waveguide_height)
    sim.set("index", n_si)
    
    # 输出波导2 (下方) - 与输入波导对齐后逐渐分开
    sim.addrect()
    sim.set("name", "output_waveguide_2")
    sim.set("x", sim_length/2 - 1e-6)
    sim.set("x span", 2e-6)
    sim.set("y", -waveguide_spacing/2)  # 最终位置
    sim.set("y span", waveguide_width)
    sim.set("z", 0)
    sim.set("z span", waveguide_height)
    sim.set("index", n_si)
    
    print("添加光源...")
    
    # 模式光源
    sim.addmode()
    sim.set("name", "source")
    sim.set("injection axis", "x-axis")
    sim.set("direction", "Forward")
    sim.set("x", -sim_length/2 + 0.5e-6)
    sim.set("y", 0)
    sim.set("z", 0)
    sim.set("y span", 2e-6)
    sim.set("z span", 2e-6)
    sim.set("wavelength start", wavelength_start)
    sim.set("wavelength stop", wavelength_stop)
    sim.set("mode selection", "fundamental TE mode")
    
    print("添加监视器...")
    
    # 优化字段监视器
    sim.addpower()
    sim.set("name", "opt_fields")
    sim.set("monitor type", "2D Z-normal")
    sim.set("x", 0)
    sim.set("x span", 2.5e-6)
    sim.set("y", 0)
    sim.set("y span", 3e-6)
    sim.set("z", 0)
    
    # FOM监视器1
    sim.addpower()
    sim.set("name", "fom_monitor_1")
    sim.set("monitor type", "2D X-normal")
    sim.set("x", sim_length/2 - 0.5e-6)
    sim.set("y", waveguide_spacing/2)
    sim.set("y span", 1.5e-6)
    sim.set("z", 0)
    sim.set("z span", 1.5e-6)
    
    # FOM监视器2
    sim.addpower()
    sim.set("name", "fom_monitor_2")
    sim.set("monitor type", "2D X-normal")
    sim.set("x", sim_length/2 - 0.5e-6)
    sim.set("y", -waveguide_spacing/2)
    sim.set("y span", 1.5e-6)
    sim.set("z", 0)
    sim.set("z span", 1.5e-6)
    
    # 网格覆盖
    sim.addmesh()
    sim.set("name", "opt_mesh")
    sim.set("x", 0)
    sim.set("x span", 2.5e-6)
    sim.set("y", 0)
    sim.set("y span", 3e-6)
    sim.set("z", 0)
    sim.set("z span", waveguide_height + 0.5e-6)
    sim.set("dx", 20e-9)
    sim.set("dy", 20e-9)
    sim.set("dz", 20e-9)
    
    print("FDTD基础仿真设置完成！")
    print("="*50)
    
    return {
        'sim_length': sim_length,
        'sim_width': sim_width,
        'sim_height': sim_height,
        'waveguide_width': waveguide_width,
        'waveguide_height': waveguide_height,
        'waveguide_spacing': waveguide_spacing,
        'n_si': n_si,
        'n_sio2': n_sio2
    }


def setup_base_simulation(sim, wavelength_start=1300e-9, wavelength_stop=1800e-9):
    """
    自动检测并设置基础仿真
    
    参数:
        sim: 仿真对象（MODE或FDTD）
        wavelength_start: 起始波长
        wavelength_stop: 终止波长
    """
    print(f"[debug] sim python type: {type(sim)}")
    # 检测是否为MODE
    is_mode = 'MODE' in str(type(sim))
    
    if is_mode:
        return setup_base_simulation_mode(sim, wavelength_start, wavelength_stop)
    else:
        return setup_base_simulation_fdtd(sim, wavelength_start, wavelength_stop)


if __name__ == "__main__":
    """测试基础设置"""
    
    print("\n选择要测试的仿真器:")
    print("1. MODE (2.5D, 快速)")
    print("2. FDTD (3D, 精确)")
    
    try:
        choice = input("\n请输入选择 (1/2): ").strip()
    except:
        choice = "1"
    
    try:
        if choice == "2":
            print("\n启动FDTD进行基础设置测试...")
            fdtd = lumapi.FDTD(hide=False)
            params = setup_base_simulation(fdtd)
            
            # 保存
            save_path = os.path.join(os.getcwd(), "y_branch_base_3d.fsp")
            fdtd.save(save_path)
            print(f"\n仿真文件已保存至: {save_path}")
            
            input("\n按回车键关闭...")
            fdtd.close()
        else:
            print("\n启动MODE进行基础设置测试...")
            mode = lumapi.MODE(hide=False)
            params = setup_base_simulation(mode)
            
            # 保存
            save_path = os.path.join(os.getcwd(), "y_branch_base_2d.lms")
            mode.save(save_path)
            print(f"\n仿真文件已保存至: {save_path}")
            
            input("\n按回车键关闭...")
            mode.close()
        
        print("\n基础仿真参数:")
        for key, value in params.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value*1e6:.3f} μm" if value < 1e-3 else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
