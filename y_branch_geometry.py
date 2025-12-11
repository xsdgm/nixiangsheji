"""
Y分支几何优化定义
定义可优化的多边形几何体
"""

import numpy as np
import scipy.interpolate as sp_interp


class YBranchGeometry:
    """Y分支可优化几何体"""
    
    def __init__(self, n_points=10):
        """
        初始化Y分支几何体
        
        参数:
            n_points: 样条节点数量
        """
        self.n_points = n_points
        
        # 定义初始节点的x坐标（固定）
        self.initial_points_x = np.linspace(-1.0e-6, 1.0e-6, n_points)
        
        # 定义初始节点的y坐标（可优化）- Y型分支设计
        # 左端(x=-1.0e-6): y=0.25e-6 (连接输入波导，宽度0.5μm)
        # 右端(x=1.0e-6): y=1.25e-6 (连接输出波导外侧，中心1.0+半宽0.25)
        self.initial_points_y = np.linspace(0.25e-6, 1.25e-6, n_points)
        
        # 定义优化边界 - 允许从0.2μm到1.5μm的范围
        self.bounds = [(0.2e-6, 1.5e-6)] * n_points
        
        # 添加约束
        # 左端不能超过输入波导宽度的一半
        self.bounds[0] = (0.24e-6, 0.28e-6)  # 左端固定在输入波导附近
        # 右端必须接近输出波导外侧
        self.bounds[-1] = (1.2e-6, 1.3e-6)  # 右端固定在输出波导外侧附近
        
        print(f"初始化Y分支几何体: {n_points}个优化参数")
        print(f"  初始形状: 左端 {self.initial_points_y[0]*1e6:.3f}μm -> 右端 {self.initial_points_y[-1]*1e6:.3f}μm")
    
    def create_polygon(self, params):
        """
        根据参数创建多边形点
        
        参数:
            params: y坐标数组，大小为n_points
            
        返回:
            polygon_points: 多边形顶点坐标数组 (N, 2)
        """
        # 1. 构建上边缘 (Outer Top)
        points_x = np.concatenate((
            [self.initial_points_x.min() - 0.1e-6],  # 输入侧延伸
            self.initial_points_x,
            [self.initial_points_x.max() + 0.1e-6]   # 输出侧延伸
        ))
        
        points_y = np.concatenate((
            [params[0]],      # 左端保持输入波导宽度
            params,
            [params[-1]]      # 右端保持输出波导位置
        ))
        
        # 使用三次样条插值创建平滑曲线
        n_interpolation_points = 100
        polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
        
        # 创建插值器
        interpolator = sp_interp.interp1d(points_x, points_y, kind='cubic')
        polygon_points_y = interpolator(polygon_points_x)
        
        # 2. 构建内边缘 (Inner Split) - V型切口
        # 分叉点在 x=0, y=0
        # 输出内侧在 x=1.0, y=0.75 (输出波导中心1.0, 宽0.5 -> 内侧0.75)
        
        x_split = 0.0e-6
        x_end = max(points_x)
        y_split = 0.0
        y_inner_end = 0.75e-6
        
        # 定义内边缘点 (线性过渡)
        inner_x = np.linspace(x_split, x_end, 50)
        inner_y = np.linspace(y_split, y_inner_end, 50)
        
        # 3. 组合多边形
        
        # 上半部分外边缘点 (从左到右)
        upper_outer = list(zip(polygon_points_x, polygon_points_y))
        
        # 上半部分内边缘点 (从右到左)
        upper_inner = list(zip(inner_x, inner_y))[::-1]
        
        # 下半部分内边缘点 (从左到右，对称)
        lower_inner = [(x, -y) for x, y in zip(inner_x, inner_y)]
        
        # 下半部分外边缘点 (从右到左，对称)
        lower_outer = [(x, -y) for x, y in upper_outer[::-1]]
        
        # 组合所有点
        final_points = []
        final_points.extend(upper_outer)
        final_points.extend(upper_inner)
        final_points.extend(lower_inner)
        final_points.extend(lower_outer)
        
        return np.array(final_points)
    
    def add_to_simulation(self, sim, params, depth=220e-9):
        """
        将优化几何体添加到仿真中
        
        参数:
            sim: FDTD/MODE仿真对象
            params: 优化参数
            depth: 几何体厚度（z方向）
        """
        # 生成多边形点
        polygon_points = self.create_polygon(params)
        
        # 检查是否已存在该结构
        try:
            sim.select("y_branch_opt")
            sim.delete()
        except:
            pass
        
        # 添加多边形
        sim.addpoly()
        sim.set("name", "y_branch_opt")
        sim.set("x", 0)
        sim.set("y", 0)
        sim.set("z", 0)
        sim.set("z span", depth)
        sim.set("vertices", polygon_points)
        sim.set("index", 3.48)  # 硅的折射率
        
        return polygon_points
    
    def visualize_shape(self, params=None):
        """
        可视化当前形状
        
        参数:
            params: 优化参数，如果为None则使用初始值
        """
        import matplotlib.pyplot as plt
        
        if params is None:
            params = self.initial_points_y
        
        polygon_points = self.create_polygon(params)
        
        plt.figure(figsize=(10, 6))
        plt.plot(polygon_points[:, 0] * 1e6, polygon_points[:, 1] * 1e6, 'b-', linewidth=2)
        plt.scatter(self.initial_points_x * 1e6, params * 1e6, c='r', s=50, zorder=5, label='Control Points')
        plt.scatter(self.initial_points_x * 1e6, -params * 1e6, c='r', s=50, zorder=5)
        plt.xlabel('X (μm)')
        plt.ylabel('Y (μm)')
        plt.title('Y-Branch Optimizable Geometry')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        return polygon_points


def export_to_gds(polygon_points, filename='y_branch_optimized.gds', layer=1):
    """
    将优化后的几何体导出为GDS文件
    
    参数:
        polygon_points: 多边形顶点坐标
        filename: 输出文件名
        layer: GDS层编号
    """
    try:
        import gdspy
        
        # 转换单位：从米到微米
        polygon_points_um = polygon_points * 1e6
        
        # 创建GDS库
        lib = gdspy.GdsLibrary()
        
        # 创建cell
        cell = lib.new_cell('Y_BRANCH')
        
        # 添加多边形
        poly = gdspy.Polygon(polygon_points_um, layer=layer)
        cell.add(poly)
        
        # 保存文件
        lib.write_gds(filename)
        print(f"GDS文件已保存至: {filename}")
        
    except ImportError:
        print("警告: 未安装gdspy库，无法导出GDS文件")
        print("可以使用以下命令安装: pip install gdspy")


if __name__ == "__main__":
    """测试几何体定义"""
    print("测试Y分支几何体定义...")
    
    # 创建几何体
    geometry = YBranchGeometry(n_points=10)
    
    print("\n初始参数:")
    print(f"  X坐标: {geometry.initial_points_x * 1e6} μm")
    print(f"  Y坐标: {geometry.initial_points_y * 1e6} μm")
    print(f"  边界: {geometry.bounds[0][0]*1e6} ~ {geometry.bounds[0][1]*1e6} μm")
    
    # 生成多边形
    polygon = geometry.create_polygon(geometry.initial_points_y)
    print(f"\n生成多边形顶点数: {len(polygon)}")
    
    # 可视化
    print("\n可视化初始形状...")
    geometry.visualize_shape()
    
    # 测试不同参数 - 展示喇叭形优化后的形状
    print("\n测试优化后的形状...")
    # 喇叭形：从0.25μm平滑过渡到1.0μm
    optimized_params = np.array([0.25, 0.30, 0.38, 0.48, 0.58, 0.68, 0.78, 0.86, 0.93, 1.00]) * 1e-6
    geometry.visualize_shape(optimized_params)
