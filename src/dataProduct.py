"""
MRI图像不均匀性仿真主程序
整合IO工具、偏场生成、噪声添加功能，提供完整的仿真流程
"""
from typing import Tuple, Optional
import numpy as np

from io_utils import read_nii_file, save_nii_file
from bias_generators import (
    generate_polynomial_bias,
    generate_gaussian_bias,
    generate_random_smooth_bias
)
from noise_utils import add_gaussian_noise


def simulate_mri_inhomogeneity(
    input_nii_path: str,
    output_nii_path: str,
    bias_type: str = "polynomial",
    bias_intensity: float = 0.3,
    polynomial_order: int = 2,
    gaussian_center: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    gaussian_sigma_ratio: float = 0.2,
    random_noise_std: float = 0.3,
    random_filter_sigma: float = 5.0,
    noise_percent: float = 5.0,
    save_bias_path: Optional[str] = None
) -> None:
    """
    磁共振图像不均匀性仿真主函数
    
    :param input_nii_path: 输入NIfTI文件路径
    :param output_nii_path: 输出NIfTI文件路径
    :param bias_type: 偏场类型（polynomial/gaussian/random_smooth）
    :param bias_intensity: 偏场强度（0-1，通用参数）
    :param polynomial_order: 多项式阶数（仅polynomial类型）
    :param gaussian_center: 高斯中心比例（仅gaussian类型）
    :param gaussian_sigma_ratio: 高斯标准差比例（仅gaussian类型）
    :param random_noise_std: 随机噪声标准差（仅random_smooth类型）
    :param random_filter_sigma: 随机滤波强度（仅random_smooth类型）
    :param noise_percent: 高斯噪声强度（百分比）
    :param save_bias_path: 偏场中间结果保存路径（None则不保存）
    """
    print("=" * 50)
    print("开始MRI图像不均匀性仿真...")
    print(f"输入文件：{input_nii_path}")
    print(f"输出文件：{output_nii_path}")
    print(f"偏场类型：{bias_type}，偏场强度：{bias_intensity}")
    print(f"噪声强度：{noise_percent}%")
    print("=" * 50)

    # 1. 读取输入数据
    try:
        data, affine = read_nii_file(input_nii_path)
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return
    field_shape = data.shape
    print(f"✅ 读取成功 - 数据形状：{field_shape}，数据范围：[{data.min():.2f}, {data.max():.2f}]")

    # 2. 生成偏场
    try:
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
            raise ValueError(f"不支持的偏场类型: {bias_type}，可选：polynomial/gaussian/random_smooth")
    except Exception as e:
        print(f"生成偏场失败：{str(e)}")
        return
    print(f"✅ 偏场生成成功 - 偏场范围：[{bias.min():.4f}, {bias.max():.4f}]")

    # 3. 保存偏场中间结果（可选）
    if save_bias_path:
        try:
            save_nii_file(bias, affine, save_bias_path)
            print(f"✅ 偏场中间结果已保存至：{save_bias_path}")
        except Exception as e:
            print(f"⚠️  保存偏场失败：{str(e)}")

    # 4. 应用偏场和噪声
    try:
        biased_data = data * bias  # 叠加偏场
        final_data = add_gaussian_noise(biased_data, noise_percent=noise_percent)  # 叠加噪声
    except Exception as e:
        print(f"处理图像失败：{str(e)}")
        return
    print(f"✅ 图像处理完成 - 最终数据范围：[{final_data.min():.2f}, {final_data.max():.2f}]")

    # 5. 保存最终结果
    try:
        save_nii_file(final_data, affine, output_nii_path)
    except Exception as e:
        print(f"保存结果失败：{str(e)}")
        return
    print("=" * 50)
    print("🎉 仿真完成！所有操作已成功执行")
    print("=" * 50)


# 测试代码（直接运行该文件时执行）
if __name__ == "__main__":
    # 示例参数配置
    INPUT_PATH = "input.nii.gz"       # 替换为你的输入NIfTI文件路径
    OUTPUT_PATH = "output_inhomogeneity.nii.gz"  # 输出文件路径
    SAVE_BIAS_PATH = "generated_bias.nii.gz"  # 偏场中间结果路径（可选）

    # 调用仿真函数
    simulate_mri_inhomogeneity(
        input_nii_path=INPUT_PATH,
        output_nii_path=OUTPUT_PATH,
        bias_type="polynomial",  # 可选：polynomial/gaussian/random_smooth
        bias_intensity=0.3,
        polynomial_order=2,
        noise_percent=5.0,
        save_bias_path=SAVE_BIAS_PATH
    )