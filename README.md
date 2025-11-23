# MRI Inhomogeneity Simulator

# contact information
Should you require a dataset， please feel free to contact me.::::lmy15933285944@163.com


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

# 磁共振不均匀性数据仿真

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
├── examples/                 # 示例脚本
├── docs/                     # 文档目录
│   ├── paper/                # 参考文献
├── dataset/                  # 数据集
│   ├── sequence1/ 
│   ├── sequence2/
│   ├── ......
│   ├── sequenceN/                   
├── requirements.txt          # 依赖清单
├── LICENSE                   # 开源许可证
└── README.md                 # 仓库首页文档（中英文简介+使用说明）
