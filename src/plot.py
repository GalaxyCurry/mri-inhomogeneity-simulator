import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------- 1. 配置文件路径和分组（新增GT数据） ----------------------
# 按模态分组：每个模态包含3个样本 + 1个GT
data_groups = {
    "T1": {
        "samples": ["E:/IXI DATA/data/1.nii.gz", "E:/IXI DATA/data/2.nii.gz", "E:/IXI DATA/data/3.nii.gz"],  # T1样本
        "gt": "E:/IXI DATA/data/T1.nii.gz"                                 # T1对应的GT
    },
    "T2": {
        "samples": ["E:/IXI DATA/data/11.nii.gz", "E:/IXI DATA/data/22.nii.gz", "E:/IXI DATA/data/33.nii.gz"],# T2样本
        "gt": "E:/IXI DATA/data/T2.nii.gz"                                 # T2对应的GT
    },
    "PD": {
        "samples": ["E:/IXI DATA/data/111.nii.gz", "E:/IXI DATA/data/222.nii.gz", "E:/IXI DATA/data/333.nii.gz"],# PD样本
        "gt": "E:/IXI DATA/data/PD.nii.gz"                                 # PD对应的GT
    }
}

# 图像保存路径（可修改）
save_path = "nifti_middle_slice_with_gt_comparison.png"

# ---------------------- 2. 读取数据并提取中间层（包含GT） ----------------------
middle_slices = {}
modalities = list(data_groups.keys())
n_samples = 3  # 每个模态的样本数

for modality in modalities:
    # 读取3个样本的中间层
    sample_slices = []
    for file_name in data_groups[modality]["samples"]:
        img = nib.load(file_name)
        data = img.get_fdata()
        z_mid = data.shape[2] // 2  # 深度方向中间层
        sample_slices.append(data[:, :, z_mid])
    
    # 读取当前模态对应的GT中间层
    gt_file = data_groups[modality]["gt"]
    gt_img = nib.load(gt_file)
    gt_data = gt_img.get_fdata()
    gt_z_mid = gt_data.shape[2] // 2  # GT的深度中间层（兼容与样本不同深度的情况）
    gt_slice = gt_data[:, :, gt_z_mid]
    
    # 存储：样本切片 + GT切片
    middle_slices[modality] = {
        "samples": sample_slices,
        "gt": gt_slice
    }

# ---------------------- 3. 绘制对比图（3行×4列：样本1-3 + GT） ----------------------
plt.rcParams['font.sans-serif'] = ['Arial']
fig, axes = plt.subplots(
    nrows=3, ncols=4,  # 3模态 × 3样本+1GT
    figsize=(18, 12),  # 加宽图像以适应4列
    tight_layout=True
)

cmap = "gray"  # 医学影像常用灰度图，可修改为其他colormap

# 遍历每个模态和列（样本1-3 + GT）
for row, modality in enumerate(modalities):
    # 绘制3个样本
    for col in range(3):
        ax = axes[row, col]
        slice_data = middle_slices[modality]["samples"][col]
        file_name = data_groups[modality]["samples"][col]
        
        # 绘制切片（优化显示范围）
        im = ax.imshow(
            slice_data.T,
            cmap=cmap,
            vmin=np.percentile(slice_data, 1),
            vmax=np.percentile(slice_data, 99)
        )
        
        # 样本子图标题
        ax.set_title(f"{modality} - {file_name}", fontsize=11, pad=8)
        ax.axis("off")
    
    # 绘制当前模态的GT（第4列）
    ax_gt = axes[row, 3]
    gt_data = middle_slices[modality]["gt"]
    gt_im = ax_gt.imshow(
        gt_data.T,
        cmap=cmap,
        vmin=np.percentile(gt_data, 1),
        vmax=np.percentile(gt_data, 99)
    )
    
    # GT子图标题（突出显示GT）
    ax_gt.set_title(f"{modality} - GT\n({data_groups[modality]['gt']})", fontsize=11, pad=8, color="#d32f2f")
    ax_gt.axis("off")
    
    # 为GT列添加颜色条（每行1个颜色条，统一强度标尺）
    cbar = plt.colorbar(gt_im, ax=ax_gt, shrink=0.8)
    cbar.set_label("Intensity", fontsize=10)

# 设置行标题（模态名称，左对齐）
for row, modality in enumerate(modalities):
    axes[row, 0].set_ylabel(modality, fontsize=14, rotation=0, ha="right", va="center", labelpad=20)

# 设置列标题（样本/GT标识，居中）
col_titles = ["Sample 1", "Sample 2", "Sample 3", "Ground Truth"]
for col, title in enumerate(col_titles):
    axes[0, col].set_xlabel(title, fontsize=14, labelpad=10)
    # GT列标题标红，突出区分
    if col == 3:
        axes[0, col].xaxis.label.set_color("#d32f2f")

# 保存高分辨率图像
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"包含GT的对比图已保存至：{Path(save_path).absolute()}")

# 显示图像（可选）
plt.show()