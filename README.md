# MRI Inhomogeneity Simulator

A lightweight, flexible tool for simulating **MRI scan data inhomogeneity** , with a focus on realistic polynomial-based bias field generation. 

This tool is designed to help researchers/engineers:
- Generate synthetic MRI data with device-specific inhomogeneity (mimicking real scanner artifacts)
- Customize inhomogeneity complexity (order, intensity, smoothness)
- Integrate with medical image simulation pipelines (compatible with NumPy/SciPy/PyTorch)

## Key Features
✅ Adaptive coefficient decay & cross-term weakening for realistic smoothness  
✅ Configurable intensity, smoothness, and polynomial parameters  
✅ 3D volume support (H×W×D) with strict dimension matching  
✅ Lightweight dependency (only NumPy + SciPy)  
✅ Easy integration with medical imaging workflows  

# 磁共振不均匀性数据仿真工具

一款轻量、灵活的MRI扫描数据不均匀性仿真代码，专注于生成真实的多项式偏场。

该工具适用于：
- 生成带设备特异性不均匀性的合成MRI数据（模拟真实扫描仪伪影）
- 自定义不均匀性复杂度（阶数、强度、光滑度）
- 集成到医学影像仿真流水线（兼容NumPy/SciPy/PyTorch）

## 核心特性 
✅ 系数自适应衰减+交叉项弱化，保证真实光滑性  
✅ 强度、光滑度、多项式参数可灵活配置  
✅ 支持3D体数据（H×W×D），维度严格匹配无错误  
✅ 轻量依赖（仅需NumPy + SciPy）  
✅ 易于集成到医学影像工作流  


<!-- GitHub Topics (add to .github/topics.txt or directly in README) -->
**Keywords**: MRI, Bias Field, Inhomogeneity Simulation, Medical Image Processing, Polynomial Bias, B0 Inhomogeneity, B1+ Bias, Image Artifact Simulation


mri-inhomogeneity-simulator/
├── src/                      # 核心代码目录
│   ├── __init__.py           # 包初始化（导出核心函数）
│   ├── bias_field.py         # 偏场生成核心代码（优化后的generate_polynomial_bias）
│   └── extensions/           # 扩展功能（未来可添加其他偏场模型）
│       ├── __init__.py
│       └── physical_bias.py  # 物理模型偏场（如B0不均匀性，可选扩展）
├── examples/                 # 示例脚本（快速上手）
│   ├── basic_polynomial_bias.py  # 多项式偏场基础使用
│   ├── 6th_order_bias_demo.py    # 6阶偏场生成示例
│   ├── integrate_with_mri_data.py # 与真实MRI数据结合示例
│   └── visualization.py      # 偏场可视化脚本（matplotlib/plotly）
├── docs/                     # 文档目录
│   ├── parameters.md         # 参数详细说明
│   ├── usage_guide.md        # 使用指南
│   └── theory.md             # 偏场仿真理论基础（可选）
├── tests/                    # 单元测试
│   ├── __init__.py
│   └── test_bias_field.py    # 偏场生成正确性测试
├── requirements.txt          # 依赖清单
├── LICENSE                   # 开源许可证
└── README.md                 # 仓库首页文档（中英文简介+使用说明）
