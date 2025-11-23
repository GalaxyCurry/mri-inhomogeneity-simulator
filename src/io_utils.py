import os
import numpy as np
import nibabel as nib
from typing import Tuple


def read_nii_file(nii_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取NIfTI文件并返回数据和仿射矩阵
    
    :param nii_path: NIfTI文件路径
    :return: 数据矩阵(shape=(H, W, D))和仿射矩阵(shape=(4,4))
    :raises FileNotFoundError: 当文件路径不存在时
    :raises ValueError: 当文件不是有效的NIfTI格式时
    """
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"NIfTI文件不存在: {nii_path}")
    try:
        img = nib.load(nii_path)
        data = img.get_fdata(dtype=np.float32)  # 强制float32类型，统一精度
        affine = img.affine.astype(np.float32)
        if data.ndim != 3:
            raise ValueError(f"仅支持3D NIfTI文件，输入文件维度为: {data.ndim}")
        return data, affine
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"无效的NIfTI文件: {str(e)}") from e


def save_nii_file(data: np.ndarray, affine: np.ndarray, save_path: str) -> None:
    """
    保存数据为NIfTI文件
    
    :param data: 待保存数据(shape=(H, W, D), dtype=float32)
    :param affine: 仿射矩阵(shape=(4,4))
    :param save_path: 保存路径
    :raises IOError: 当保存路径不可写时
    :raises ValueError: 当数据维度或类型不合法时
    """
    
    if data.ndim != 3:
        raise ValueError(f"数据必须为3D，当前维度: {data.ndim}")
    if affine.shape != (4, 4):
        raise ValueError(f"仿射矩阵必须为(4,4)，当前形状: {affine.shape}")
    
    # 确保数据类型为float32
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        img = nib.Nifti1Image(data, affine)
        nib.save(img, save_path)
        print(f"文件已保存至：{save_path}")
    except PermissionError:
        raise IOError(f"无权限写入文件: {save_path}")
    except Exception as e:
        raise IOError(f"保存文件失败: {str(e)}") from e