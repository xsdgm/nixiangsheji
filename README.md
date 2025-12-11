# Y分支光子逆向设计 - Lumerical Python API

本项目复现了Ansys Lumerical的光子逆向设计教程，实现了Y分支波导分束器的自动优化设计。

## 项目概述

本项目展示如何使用Lumerical FDTD/MODE求解器和Python API进行光子器件的逆向设计（Inverse Design）。通过伴随方法（Adjoint Method）和梯度优化算法，自动找到最优的Y分支几何形状以实现最大化的传输效率。

## 文件说明

### 核心文件

1. **y_branch_lumopt.py**
   - 完整的lumopt优化脚本
   - 使用官方lumopt库和伴随方法
   - 推荐的主优化脚本

2. **run_optimization.py**
   - 启动脚本，检查依赖并运行lumopt优化

3. **monitor_optimization.py**
   - 优化过程监控工具
   - 支持实时监控、文件列表、状态查看

4. **y_branch_base_setup.py**
   - 基础仿真环境设置
   - 定义仿真区域、材料、光源、监视器等

5. **y_branch_geometry.py**
   - 可优化几何体定义
   - 样条插值创建平滑的Y分支形状
   - 包含可视化和GDS导出功能## 使用方法

### 运行优化

启动完整的lumopt优化：

```bash
python run_optimization.py
```

或直接运行lumopt脚本：

```bash
python y_branch_lumopt.py
```

脚本会自动检查所有依赖（lumapi、lumopt、numpy、scipy、matplotlib），然后启动伴随方法优化。

### 监控优化过程

优化运行期间，在另一个终端窗口使用监控工具：

**查看当前优化状态：**
```bash
python monitor_optimization.py
```

**实时监控模式（每10秒更新）：**
```bash
python monitor_optimization.py --watch
```

**查看最新生成的文件：**
```bash
python monitor_optimization.py --latest
```

**监控指定的优化文件夹：**
```bash
python monitor_optimization.py opts_001
```

## 依赖库

### 必需（Python标准库或常用库）
- numpy
- scipy
- matplotlib

### 可选
- gdspy（用于GDS文件导出）
  ```bash
  pip install gdspy
  ```

### Lumerical相关
- lumapi（随Lumerical安装）
- lumopt（随Lumerical安装，部分版本可能没有）

## 设计参数

### 波导参数
- 波导宽度: 500 nm
- 波导高度: 220 nm (SOI标准)
- 输出波导间距: 2 μm

### 材料参数
- 硅折射率: 3.48 (@1550nm)
- 二氧化硅折射率: 1.44 (@1550nm)

### 优化参数
- 样条节点数: 10
- Y坐标范围: 0.2 ~ 0.8 μm
- 波长范围: 1300 ~ 1800 nm

### 优化目标
最大化两个输出端口的总传输效率

## 工作原理

### 1. 伴随方法（Adjoint Method）
- 仅需两次仿真（正向+伴随）即可计算梯度
- 无论优化参数数量多少，计算成本恒定
- 比有限差分法效率提高N倍（N为参数数量）

### 2. 优化流程
1. 定义初始Y分支几何形状（线性锥形）
2. 运行正向仿真，计算传输效率（FOM）
3. 运行伴随仿真，计算梯度
4. 使用梯度信息更新几何参数
5. 重复2-4直到收敛

### 3. 几何参数化
- 使用样条曲线定义Y分支边界
- 控制点的y坐标为优化变量
- 三次样条插值保证平滑性

## 输出结果

### 文件输出
- `y_branch_optimized_params.txt` - 优化后的参数
- `y_branch_optimized.gds` - GDS版图文件
- `optimization_history.png` - 优化历史图
- `y_branch_base.lms` - 基础仿真文件（MODE）

### 可视化
- FOM演化曲线
- 参数演化曲线
- 几何形状对比图

## 预期结果

根据Ansys官方教程：
- 2.5D优化（varFDTD）: 传输效率 ~98%
- 3D优化（FDTD）: 传输效率 ~95%
- 优化迭代次数: 20-30次
- 单次迭代时间: 2D ~1分钟，3D ~10-30分钟

## 故障排除

### 问题1: 无法导入lumapi
**解决方案**: 
- 检查Lumerical安装路径
- 修改脚本开头的 `lumerical_path` 变量
- 确保Python版本兼容（通常需要Python 3.7-3.9）

### 问题2: 未找到lumopt
**解决方案**:
- lumopt在某些Lumerical版本中可能不包含
- 使用 `y_branch_optimization.py` 代替，它不依赖lumopt
- 或从GitHub下载: https://github.com/chriskeraly/lumopt

### 问题3: 许可证错误
**解决方案**:
- 确保FDTD/MODE许可证有效
- varFDTD需要单独的许可证
- API许可证可能需要单独配置

### 问题4: 优化不收敛
**解决方案**:
- 调整初始参数（更接近合理形状）
- 减小参数边界范围
- 增加网格精度
- 尝试不同的优化器（BFGS, CG等）

## 进阶定制

### 修改波导参数
编辑 `y_branch_base_setup.py` 中的参数（同时支持MODE和FDTD）：
```python
waveguide_width = 0.5e-6   # 修改波导宽度
waveguide_height = 0.22e-6  # 修改波导高度
waveguide_spacing = 2e-6    # 修改输出间距
```

### 修改优化参数
编辑 `y_branch_geometry.py`:
```python
n_points = 15  # 增加控制点数量
bounds = [(0.15e-6, 0.9e-6)]  # 修改y坐标范围
```

### 修改波长范围
编辑优化脚本：
```python
wavelength_start = 1500e-9  # 起始波长
wavelength_stop = 1600e-9   # 终止波长
```

## 参考资料

### 官方文档
- [Photonic Inverse Design Overview](https://optics.ansys.com/hc/en-us/articles/360049853854)
- [Inverse Design of Y-branch](https://optics.ansys.com/hc/en-us/articles/360042305274)
- [lumopt Documentation](https://lumopt.readthedocs.io/)

### 学术论文
- Hughes et al., "Adjoint Method and Inverse Design for Nonlinear Nanophotonic Devices", ACS Photonics (2018)
- Lalau-Keraly et al., "Adjoint shape optimization applied to electromagnetic design", Optics Express (2013)

## 许可证

本代码仅用于学习和研究目的。使用Lumerical软件需要相应的商业许可证。

## 作者

根据Ansys Lumerical官方教程改编实现

## 版本历史

- v1.0 (2025-12-09): 初始版本
  - 基础仿真设置
  - 几何体定义
  - 简化优化实现
  - lumopt集成
