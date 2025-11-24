import nibabel as nib
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal


def read_nii_file(nii_path):
    
    img = nib.load(nii_path)
    data = img.get_fdata()  
    affine = img.affine  
    return data, affine

def save_nii_file(data, affine, save_path):

    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, save_path)
    print(f"仿真结果已保存至：{save_path}")










def generate_polynomial_bias(
    field_shape: Tuple[int, int, int],
    order: int = 2,
    intensity: float = 0.3,
    base_coeff: float = 0.3,
    decay_factor: float = 0.5,
    cross_factor: float = 0.8
) -> np.ndarray:
    """
    生成多项式曲面偏场（模拟设备固有偏场）- 优化版（支持2-6阶完整项）
    
    核心优化：
    - 自动生成2-6阶所有多项式项（含全部交叉项，如3阶的x²y、xyz等）
    - 系数随阶数衰减，交叉项强度自适应弱化，保证偏场光滑
    - 坐标维度严格匹配输入形状，避免维度错误
    - 高斯滤波sigma自适应输入尺寸，兼顾不同大小的偏场
    - 完善输入验证，参数可灵活调整
    
    :param field_shape: 偏场形状 (H, W, D)，对应高度、宽度、深度
    :param order: 多项式阶数（1-6，重点优化2-6阶）
    :param intensity: 偏场整体强度（0-1，越大不均匀越明显）
    :param base_coeff: 1阶项基础系数（控制低阶项贡献度）
    :param decay_factor: 阶数衰减因子（0-1，越小高阶项衰减越快）
    :param cross_factor: 交叉项系数因子（0-1，控制交叉项相对于单项的强度）
    :return: 偏场矩阵（值范围1-intensity~1+intensity，shape=(H, W, D)）
    """
    # 1. 输入参数验证（避免无效输入）
    if len(field_shape) != 3:
        raise ValueError("field_shape必须是3维元组 (H, W, D)")
    H, W, D = field_shape
    if not (1 <= order <= 6):
        raise ValueError("order必须在1-6之间（优化重点支持2-6阶）")
    if base_coeff <= 0 or decay_factor <= 0 or decay_factor >= 1 or cross_factor <= 0:
        raise ValueError("base_coeff>0、0<decay_factor<1、cross_factor>0")
    
    # 2. 生成归一化坐标（-1~1），确保维度与field_shape一致
    x = np.linspace(-1, 1, W, dtype=np.float32)  # x轴对应宽度W（第二维）
    y = np.linspace(-1, 1, H, dtype=np.float32)  # y轴对应高度H（第一维）
    z = np.linspace(-1, 1, D, dtype=np.float32)  # z轴对应深度D（第三维）
    # indexing='xy'保证输出shape=(H, W, D)，与field_shape严格匹配
    xx, yy, zz = np.meshgrid(x, y, z, indexing='xy')
    
    # 3. 自动生成所有多项式项（a+b+c=n，n从1到order）
    polynomial_terms = []
    for n in range(1, order + 1):  # 遍历1~order阶
        for a in range(n + 1):      # x的指数a（0~n）
            for b in range(n - a + 1):  # y的指数b（0~n-a）
                c = n - a - b       # z的指数c（由a+b+c=n推导）
                polynomial_terms.append((a, b, c, n))  # 存储(a,b,c,阶数n)
    
    # 4. 初始化偏场并累加所有多项式项
    bias = np.ones(field_shape, dtype=np.float32)  # 基础值=1（无偏场）
    for a, b, c, n in polynomial_terms:
        # 计算当前项：x^a * y^b * z^c
        term = (xx ** a) * (yy ** b) * (zz ** c)
        
        # 计算自适应系数：高阶衰减+交叉项弱化
        coeff = base_coeff * (decay_factor ** (n - 1))  # 阶数衰减（n越高系数越小）
        # 判断是否为单项（如x²、y³，无交叉），交叉项系数乘以cross_factor弱化
        is_single_term = (a == n and b == 0 and c == 0) or \
                         (b == n and a == 0 and c == 0) or \
                         (c == n and a == 0 and b == 0)
        if not is_single_term:
            coeff *= cross_factor
        
        # 累加当前项（乘以强度intensity控制整体影响）
        bias += intensity * coeff * term
    
    # 5. 限制偏场范围+自适应高斯平滑
    bias = np.clip(bias, 1 - intensity, 1 + intensity)  # 避免过暗/过亮
    # sigma自适应输入尺寸（最小维度的3%，限制在1-5之间，保证平滑效果）
    min_dim = min(field_shape)
    sigma = max(1.0, min(5.0, min_dim * 0.03))
    bias = gaussian_filter(bias, sigma=sigma)
    
    return bias













def generate_gaussian_bias(field_shape, center_ratio=(0.5, 0.5, 0.5), sigma_ratio=0.2, intensity=0.4):
    """
    生成高斯分布偏场（模拟表面线圈局部强信号）
    :param field_shape: 偏场形状 (H, W, D)
    :param center_ratio: 高斯核中心（相对于图像尺寸的比例，0-1）
    :param sigma_ratio: 高斯核标准差（相对于图像尺寸的比例，0.1-0.3）
    :param intensity: 偏场强度（0-1，越大中心与边缘差异越明显）
    :return: 偏场矩阵（中心强、边缘弱，光滑变化）
    """
    H, W, D = field_shape
    # 计算高斯核中心（像素坐标）
    center = (
        center_ratio[0] * H,
        center_ratio[1] * W,
        center_ratio[2] * D
    )
    # 计算高斯核标准差（像素单位）
    sigma = (
        sigma_ratio * H,
        sigma_ratio * W,
        sigma_ratio * D
    )

    # 生成三维高斯分布
    x = np.linspace(0, H - 1, H)
    y = np.linspace(0, W - 1, W)
    z = np.linspace(0, D - 1, D)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # 关键修改：使用stack代替dstack，确保坐标形状为(H, W, D, 3)
    pos = np.stack((xx, yy, zz), axis=-1)  # 改为stack并指定axis=-1

    # 多元高斯分布概率密度（归一化到0-1）
    gaussian = multivariate_normal.pdf(pos, mean=center, cov=np.diag(sigma) **2)
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

    # 构建偏场（中心1+intensity，边缘1-intensity）
    bias = 1 - intensity + 2 * intensity * gaussian
    # 高斯滤波平滑
    bias = gaussian_filter(bias, sigma=2)
    return bias










def generate_random_smooth_bias(field_shape, noise_std=0.3, filter_sigma=5, intensity=0.3):
    """
    生成随机光滑偏场（模拟复杂设备误差）
    :param field_shape: 偏场形状 (H, W, D)
    :param noise_std: 初始随机噪声标准差（0.2-0.5）
    :param filter_sigma: 高斯滤波标准差（越大越光滑，3-7）
    :param intensity: 偏场强度（0-1）
    :return: 偏场矩阵（随机光滑变化）
    """
    # 生成随机噪声
    np.random.seed(42)  # 固定种子保证可复现
    random_noise = np.random.normal(loc=0, scale=noise_std, size=field_shape)
    # 低通滤波得到光滑噪声（即偏场）
    smooth_noise = gaussian_filter(random_noise, sigma=filter_sigma)
    # 归一化并缩放至目标强度
    smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    bias = 1 - intensity + 2 * intensity * smooth_noise
    return bias












def add_gaussian_noise(data, noise_percent=5):
    """
    添加高斯噪声（模拟磁共振图像噪声）
    :param data: 输入图像数据
    :param noise_percent: 噪声强度（相对于数据最大值的百分比，3-10为宜）
    :return: 带噪声的图像数据
    """
    np.random.seed(42)
    max_val = data.max()
    noise_std = max_val * (noise_percent / 100)
    noise = np.random.normal(loc=0, scale=noise_std, size=data.shape)
    # 噪声叠加后裁剪到原始数据范围（避免负值和溢出）
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, data.min(), max_val)
    return noisy_data





def simulate_mri_inhomogeneity(
        input_nii_path,
        output_nii_path,
        bias_type="polynomial",  # 可选：polynomial/gaussian/random_smooth
        bias_intensity=0.3,
        polynomial_order=2,
        gaussian_center=(0.5, 0.5, 0.5),
        gaussian_sigma_ratio=0.2,
        random_noise_std=0.3,
        random_filter_sigma=5,
        noise_percent=5
):
    """
    磁共振图像不均匀性仿真主函数
    :param input_nii_path: 输入nii文件路径
    :param output_nii_path: 输出nii文件路径
    :param bias_type: 偏场类型
    :param bias_intensity: 偏场强度（0-1，所有偏场通用）
    :param polynomial_order: 多项式偏场阶数（仅polynomial类型有效）
    :param gaussian_center: 高斯偏场中心（仅gaussian类型有效）
    :param gaussian_sigma_ratio: 高斯偏场标准差比例（仅gaussian类型有效）
    :param random_noise_std: 随机偏场初始噪声标准差（仅random_smooth类型有效）
    :param random_filter_sigma: 随机偏场滤波标准差（仅random_smooth类型有效）
    :param noise_percent: 噪声强度（百分比）
    """
    data, affine = read_nii_file(input_nii_path)
    print(f"原始数据形状：{data.shape}")

    field_shape = data.shape
    if bias_type == "polynomial":
        bias = generate_polynomial_bias(
            field_shape=field_shape,
            order=polynomial_order,
            intensity=bias_intensity
        )
    elif bias_type == "gaussian":
        bias = generate_gaussian_bias(
            field_shape=field_shape,
            center_ratio=gaussian_center,
            sigma_ratio=gaussian_sigma_ratio,
            intensity=bias_intensity
        )
    elif bias_type == "random_smooth":
        bias = generate_random_smooth_bias(
            field_shape=field_shape,
            noise_std=random_noise_std,
            filter_sigma=random_filter_sigma,
            intensity=bias_intensity
        )
    else:
        raise ValueError("bias_type仅支持：polynomial/gaussian/random_smooth")

    biased_data = data * bias

    final_data = add_gaussian_noise(biased_data, noise_percent=noise_percent)

    save_nii_file(final_data, affine, output_nii_path)

# ------------- 示例调用 -------------
if __name__ == "__main__":
    
    INPUT_NII_PATH = "E:/IXI DATA/IXI-PD/IXI013-HH-1212-PD.nii.gz"  # 输入nii文件路径
    OUTPUT_NII_PATH = "E:/IXI DATA/333.nii.gz"  # 输出nii文件路径
    BIAS_TYPE = "random_smooth"  # 偏场类型：polynomial/gaussian/random_smooth
    BIAS_INTENSITY = 2  # 偏场强度（0-1，越大不均匀越明显）
    NOISE_PERCENT = 2  # 噪声强度（3-10为宜）

    # 多项式偏场专属参数（仅BIAS_TYPE="polynomial"时生效）
    POLYNOMIAL_ORDER = 4  # 多项式阶数（2=二次，4=四次）

    # 高斯偏场专属参数（仅BIAS_TYPE="gaussian"时生效）
    GAUSSIAN_CENTER = (0.5, 0.5, 0.5)  # 偏场中心（图像中心比例，如(0.3,0.5,0.5)表示x方向30%处）
    GAUSSIAN_SIGMA_RATIO = 0.4  # 高斯核大小比例（0.1-0.3）

    # 随机光滑偏场专属参数（仅BIAS_TYPE="random_smooth"时生效）
    RANDOM_NOISE_STD = 2  # 初始噪声标准差
    RANDOM_FILTER_SIGMA = 8  # 平滑滤波强度（越大越光滑）
    # -----------------------------------------------------------------------------

    
    simulate_mri_inhomogeneity(
        input_nii_path=INPUT_NII_PATH,
        output_nii_path=OUTPUT_NII_PATH,
        bias_type=BIAS_TYPE,
        bias_intensity=BIAS_INTENSITY,
        polynomial_order=POLYNOMIAL_ORDER,
        gaussian_center=GAUSSIAN_CENTER,
        gaussian_sigma_ratio=GAUSSIAN_SIGMA_RATIO,
        random_noise_std=RANDOM_NOISE_STD,
        random_filter_sigma=RANDOM_FILTER_SIGMA,
        noise_percent=NOISE_PERCENT
    )
    print("仿真完成！")