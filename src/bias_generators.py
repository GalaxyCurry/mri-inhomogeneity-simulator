"""
MRI偏场生成工具类
提供多项式、高斯、随机光滑三种偏场生成方法，模拟设备不均匀性
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import product
from typing import Tuple, Optional


def generate_polynomial_bias(
    field_shape: Tuple[int, int, int],
    order: int = 2,
    intensity: float = 0.3,
    base_coeff: float = 0.3,
    decay_factor: float = 0.5,
    cross_factor: float = 0.8,
    filter_sigma: Optional[float] = None
) -> np.ndarray:
    """
    生成多项式曲面偏场（模拟设备固有偏场）
    
    :param field_shape: 偏场形状 (H, W, D)
    :param order: 多项式阶数（1-6）
    :param intensity: 偏场整体强度（0-1）
    :param base_coeff: 1阶项基础系数（>0）
    :param decay_factor: 阶数衰减因子（0-1）
    :param cross_factor: 交叉项系数因子（0-1）
    :param filter_sigma: 高斯滤波标准差（None则自适应计算）
    :return: 偏场矩阵(shape=(H, W, D), 1-intensity~1+intensity)
    """
    # 严格参数验证
    if not isinstance(field_shape, tuple) or len(field_shape) != 3:
        raise ValueError(f"field_shape必须是3元组，当前: {field_shape}")
    H, W, D = field_shape
    for dim in (H, W, D):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"维度必须为正整数，当前: {field_shape}")
    if not isinstance(order, int) or not (1 <= order <= 6):
        raise ValueError(f"order必须是1-6的整数，当前: {order}")
    for param, name in zip(
        [intensity, base_coeff, decay_factor, cross_factor],
        ["intensity", "base_coeff", "decay_factor", "cross_factor"]
    ):
        if not isinstance(param, (int, float)) or param < 0:
            raise ValueError(f"{name}必须为非负数，当前: {param}")
    if not (0 < decay_factor < 1):
        raise ValueError(f"decay_factor必须在(0,1)之间，当前: {decay_factor}")

    # 生成归一化坐标（-1~1），明确维度对应关系
    x = np.linspace(-1, 1, W, dtype=np.float32)  # x → 宽度（第二维）
    y = np.linspace(-1, 1, H, dtype=np.float32)  # y → 高度（第一维）
    z = np.linspace(-1, 1, D, dtype=np.float32)  # z → 深度（第三维）
    xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')  # 输出shape=(H, W, D)

    # 优化多项式项生成（使用itertools替代多重循环，更高效）
    polynomial_terms = []
    for n in range(1, order + 1):
        # 生成a+b+c=n的所有非负整数解（a:x指数, b:y指数, c:z指数）
        for a, b in product(range(n + 1), range(n + 1 - a)):
            c = n - a - b
            polynomial_terms.append((a, b, c, n))

    # 累加多项式项（使用向量化操作减少循环开销）
    bias = np.ones(field_shape, dtype=np.float32)
    for a, b, c, n in polynomial_terms:
        term = (xx ** a) * (yy ** b) * (zz ** c)
        # 系数计算（保持原逻辑，增加可读性注释）
        coeff = base_coeff * (decay_factor ** (n - 1))  # 高阶项衰减
        if not (a == n and b == 0 and c == 0) and \
           not (b == n and a == 0 and c == 0) and \
           not (c == n and a == 0 and b == 0):
            coeff *= cross_factor  # 交叉项弱化
        bias += intensity * coeff * term

    # 限制范围并平滑
    bias = np.clip(bias, 1 - intensity, 1 + intensity)
    # 自适应或手动滤波sigma
    if filter_sigma is None:
        min_dim = min(field_shape)
        filter_sigma = max(1.0, min(5.0, min_dim * 0.03))
    bias = gaussian_filter(bias, sigma=filter_sigma)

    return bias


def generate_gaussian_bias(
    field_shape: Tuple[int, int, int],
    center_ratio: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    sigma_ratio: float = 0.2,
    intensity: float = 0.4,
    filter_sigma: float = 2.0
) -> np.ndarray:
    """
    生成高斯分布偏场（模拟表面线圈局部强信号）
    
    :param field_shape: 偏场形状 (H, W, D)
    :param center_ratio: 高斯中心（相对于图像尺寸的比例，0-1）
    :param sigma_ratio: 高斯标准差比例（0.1-0.3为宜）
    :param intensity: 偏场强度（0-1）
    :param filter_sigma: 高斯滤波标准差（控制平滑度）
    :return: 偏场矩阵(shape=(H, W, D))
    """
    H, W, D = field_shape
    # 验证中心坐标在有效范围
    for i, (ratio, dim_name) in enumerate(zip(center_ratio, ["H", "W", "D"])):
        if not (0 <= ratio <= 1):
            raise ValueError(f"center_ratio[{i}]必须在[0,1]之间，当前: {ratio}")
    if not (0 < sigma_ratio < 1):
        raise ValueError(f"sigma_ratio必须在(0,1)之间，当前: {sigma_ratio}")

    # 计算中心和标准差（像素坐标）
    center = (
        center_ratio[0] * (H - 1),  # 修正：使用H-1确保在有效索引内
        center_ratio[1] * (W - 1),
        center_ratio[2] * (D - 1)
    )
    sigma = (
        sigma_ratio * H,
        sigma_ratio * W,
        sigma_ratio * D
    )

    # 优化高斯分布计算（避免创建大维度pos数组，直接计算指数部分）
    x = np.linspace(0, H-1, H, dtype=np.float32)
    y = np.linspace(0, W-1, W, dtype=np.float32)
    z = np.linspace(0, D-1, D, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')  # 保持与输入形状一致

    # 直接计算高斯概率密度（避免multivariate_normal的高内存占用）
    gaussian = np.exp(-0.5 * (
        ((xx - center[0]) / sigma[0])**2 +
        ((yy - center[1]) / sigma[1])** 2 +
        ((zz - center[2]) / sigma[2])**2
    ))
    # 归一化（处理全零边缘情况）
    g_max, g_min = gaussian.max(), gaussian.min()
    if g_max - g_min < 1e-9:
        return np.ones(field_shape, dtype=np.float32)  # 无偏场
    gaussian = (gaussian - g_min) / (g_max - g_min)

    # 构建偏场并平滑
    bias = 1 - intensity + 2 * intensity * gaussian
    bias = gaussian_filter(bias, sigma=filter_sigma)
    return bias.astype(np.float32)


def generate_random_smooth_bias(
    field_shape: Tuple[int, int, int],
    noise_std: float = 0.3,
    filter_sigma: float = 5.0,
    intensity: float = 0.3,
    random_seed: Optional[int] = 42
) -> np.ndarray:
    """
    生成随机光滑偏场（模拟复杂设备误差）
    
    :param field_shape: 偏场形状 (H, W, D)
    :param noise_std: 初始随机噪声标准差（0.2-0.5为宜）
    :param filter_sigma: 高斯滤波标准差（越大越光滑，3-7为宜）
    :param intensity: 偏场强度（0-1）
    :param random_seed: 随机种子（None则不固定）
    :return: 偏场矩阵(shape=(H, W, D))
    """
    # 参数验证
    if noise_std <= 0:
        raise ValueError(f"noise_std必须为正数，当前: {noise_std}")
    if filter_sigma <= 0:
        raise ValueError(f"filter_sigma必须为正数，当前: {filter_sigma}")

    # 可控随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    # 生成随机噪声（使用float32减少内存）
    random_noise = np.random.normal(loc=0, scale=noise_std, size=field_shape).astype(np.float32)
    
    # 平滑噪声（偏场核心）
    smooth_noise = gaussian_filter(random_noise, sigma=filter_sigma)
    
    # 安全归一化（处理极端情况）
    sn_min, sn_max = smooth_noise.min(), smooth_noise.max()
    if sn_max - sn_min < 1e-9:
        return np.ones(field_shape, dtype=np.float32)
    smooth_noise = (smooth_noise - sn_min) / (sn_max - sn_min)
    
    # 缩放至目标强度
    bias = 1 - intensity + 2 * intensity * smooth_noise
    return bias.astype(np.float32)