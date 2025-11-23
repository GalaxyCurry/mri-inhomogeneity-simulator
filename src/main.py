import os
import numpy as np
from datetime import datetime

from dataProduct import simulate_mri_inhomogeneity


def batch_mri_random_simulation(root_dir):
    
    rng = np.random.default_rng()

    # 序列文件夹名字列表
    sequences = ["T1", "T2", "PD"]
    
    method_params_ranges = {
        # 方法1：多项式偏场
        "polynomial": {
            "bias_intensity": (0.2, 1.0),    # 偏场强度范围
            "polynomial_order": (2, 6),      # 多项式阶数范围（整数）
            "noise_percent": (0, 3)          # 噪声百分比范围
        },
        # 方法2：高斯偏场
        "gaussian": {
            "bias_intensity": (0.2, 1.0),    # 偏场强度范围
            "gaussian_center": [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],  # 中心坐标范围（x,y,z）
            "gaussian_sigma_ratio": (0.2, 0.4),  # sigma比例范围
            "noise_percent": (0, 3)          # 噪声百分比范围
        },
        # 方法3：随机光滑偏场
        "random_smooth": {
            "bias_intensity": (0.2, 1.0),    # 偏场强度范围
            "random_noise_std": (0.1, 0.6),  # 噪声标准差范围
            "random_filter_sigma": (2, 8),   # 滤波sigma范围（整数）
            "noise_percent": (0, 3)          # 噪声百分比范围
        }
    }
    
    # 方法列表
    methods_order = ["polynomial", "gaussian", "random_smooth"]
    
    # 遍历每个序列文件夹
    for seq in sequences:
        seq_dir = os.path.join(root_dir, seq)
        if not os.path.exists(seq_dir):
            print(f"警告：序列文件夹 {seq_dir} 不存在，跳过...")
            continue
        
        nii_files = [f for f in os.listdir(seq_dir) if f.endswith(".nii.gz")]
        nii_files.sort()  
        total_files = len(nii_files)
        
        if total_files == 0:
            print(f"警告：序列 {seq} 文件夹中没有.nii.gz文件，跳过...")
            continue
        
        print(f"\n===== 开始处理序列: {seq} =====")
        print(f"序列 {seq} 共找到 {total_files} 个文件")
        
        # 核心：3:3:1 比例分配，第三组占 1/7，剩余平分给前两组
        group3_size = total_files // 7  # 第三组：1/7（整数除法保证不超比例）
        group1_size = (total_files - group3_size) // 2  # 前两组平分剩余文件
        group2_size = total_files - group1_size - group3_size  

        groups = [
            nii_files[:group1_size],  
            nii_files[group1_size:group1_size+group2_size],  
            nii_files[-group3_size:]  
        ]
        
        print(f"分组情况：")
        print(f"  第一组（{methods_order[0]}）：{len(groups[0])} 个文件")
        print(f"  第二组（{methods_order[1]}）：{len(groups[1])} 个文件")
        print(f"  第三组（{methods_order[2]}）：{len(groups[2])} 个文件")
        
        
        output_root = os.path.join(root_dir, f"{seq}_simulated_random")
        os.makedirs(output_root, exist_ok=True)
        
        # 创建日志文件（记录每个文件的随机参数）
        log_filename = f"{seq}_simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = os.path.join(output_root, log_filename)
        
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"MRI仿真参数日志 - 序列：{seq}\n")
            log_file.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"总文件数：{total_files}\n")
            log_file.write("="*80 + "\n")
            log_file.write(f"{'文件序号':<6} {'文件名':<30} {'方法':<15} {'参数详情':<50}\n")
            log_file.write("="*80 + "\n")
        
        # 处理每组文件（对应一种方法）
        for group_idx, (method, group_files) in enumerate(zip(methods_order, groups)):
            if not group_files:
                print(f"\n第{group_idx+1}组（{method}）无文件，跳过...")
                continue
            
            print(f"\n===== 处理第{group_idx+1}组（{method}方法）- 共{len(group_files)}个文件 =====")
            
            # 创建当前方法的输出目录
            method_output_dir = os.path.join(output_root, method)
            os.makedirs(method_output_dir, exist_ok=True)
            
            
            for file_idx, filename in enumerate(group_files):
                input_path = os.path.join(seq_dir, filename)
                base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
                global_idx = sum(len(groups[i]) for i in range(group_idx)) + file_idx  # 全局文件序号
                
                print(f"\n[{global_idx+1}/{total_files}] 处理文件: {filename}")
                
                # 随机生成当前方法的参数
                params_range = method_params_ranges[method]
                random_params = {}
                
                # 生成通用参数
                random_params["bias_intensity"] = round(rng.uniform(*params_range["bias_intensity"]), 3)
                random_params["noise_percent"] = round(rng.uniform(*params_range["noise_percent"]), 3)
                
                # 生成各方法特定参数
                if method == "polynomial":
                    # 多项式阶数：整数
                    random_params["polynomial_order"] = rng.integers(*params_range["polynomial_order"])
                    
                elif method == "gaussian":
                    # 高斯中心：每个维度在对应范围内随机生成
                    gaussian_center = tuple(
                        round(rng.uniform(*params_range["gaussian_center"][dim]), 3)
                        for dim in range(3)
                    )
                    random_params["gaussian_center"] = gaussian_center
                    random_params["gaussian_sigma_ratio"] = round(rng.uniform(*params_range["gaussian_sigma_ratio"]), 3)
                    
                elif method == "random_smooth":
                    # 滤波sigma：整数
                    random_params["random_filter_sigma"] = rng.integers(*params_range["random_filter_sigma"])
                    random_params["random_noise_std"] = round(rng.uniform(*params_range["random_noise_std"]), 3)
                
                # 构造输出文件名（按原文件名输出）
                output_filename = f"{base_name}.nii"
                output_path = os.path.join(method_output_dir, output_filename)
                
    
                print(f"  随机参数：{random_params}")
                simulate_mri_inhomogeneity(
                    input_nii_path=input_path,
                    output_nii_path=output_path,
                    bias_type=method,
                    **random_params  
                )
                
                with open(log_path, "a", encoding="utf-8") as log_file:
                    param_str = ", ".join([f"{k}={v}" for k, v in random_params.items()])
                    log_file.write(f"{global_idx+1:<6} {filename:<30} {method:<15} {param_str:<50}\n")
            
                print(f"  文件 {filename} 处理完成")
        
        print(f"\n===== 序列 {seq} 处理完成 =====")
        print(f"参数日志文件：{log_path}")

if __name__ == "__main__":
    
    ROOT_DIRECTORY = "E:/IXI DATA"  
    
    batch_mri_random_simulation(ROOT_DIRECTORY)
    print("\n所有序列处理完成！")