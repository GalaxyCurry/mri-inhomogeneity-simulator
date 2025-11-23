"""
MRI图像噪声工具类
提供高斯噪声添加功能，模拟磁共振图像的随机噪声特性
"""
import numpy as np
from typing import Optional


def add_gaussian_noise(
    data: np.ndarray,
    noise_percent: float = 5.0,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """
    添加高斯噪声（模拟磁共振图像噪声）
    
    :param data: 输入图像数据(shape=(H, W, D))
    :param noise_percent: 噪声强度（相对于数据动态范围的百分比，3-10为宜）
    :param random_seed: 随机种子（None则不固定）
    :return: 带噪声的图像数据(shape与输入一致)
    """
    if noise_percent < 0:
        raise ValueError(f"noise_percent不能为负数，当前: {noise_percent}")
    if data.ndim != 3:
        raise ValueError(f"输入数据必须为3D，当前维度: {data.ndim}")

    # 计算数据动态范围（避免使用max-min=0的情况）
    data_min, data_max = data.min(), data.max()
    dynamic_range = data_max - data_min
    if dynamic_range < 1e-9:
        # 数据无变化，直接返回原数据
        return data.copy()

    # 噪声标准差计算（基于动态范围而非最大值，更合理）
    noise_std = dynamic_range * (noise_percent / 100)
    
    # 可控随机噪声
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(loc=0, scale=noise_std, size=data.shape).astype(np.float32)
    
    # 叠加噪声并限制在原始数据范围内
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, data_min, data_max)
    return noisy_data.astype(data.dtype)  # 保持与输入数据类型一致