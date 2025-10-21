import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from data_utils import read_ir_image, add_salt_pepper_noise
from noise_estimation import noise_detection
from lsq_denoise import lsq_restore

# 设置matplotlib中文字体支持
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

def adaptive_median_filter(img, max_win_size=7):
    """基线方法：真实自适应中值滤波（动态窗口，避免固定窗口过弱）"""
    img = img.astype(np.uint8)
    h, w = img.shape
    result = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            win_size = 3  # 起始窗口3×3
            while win_size <= max_win_size:
                half = win_size // 2
                # 边界处理（边缘复制）
                i_s = max(0, i - half)
                i_e = min(h, i + half + 1)
                j_s = max(0, j - half)
                j_e = min(w, j + half + 1)
                window = img[i_s:i_e, j_s:j_e]
                
                # 计算窗口统计量
                med = np.median(window)
                min_w = np.min(window)
                max_w = np.max(window)
                
                # 自适应判断
                if min_w < med < max_w:
                    if min_w < img[i, j] < max_w:
                        result[i, j] = img[i, j]
                    else:
                        result[i, j] = med
                    break
                else:
                    win_size += 2  # 扩大窗口
            else:
                result[i, j] = med  # 最大窗口仍无效，用中值
    
    return result

def fsim(original, denoised, data_range=255):
    """
    优化的FSIM计算（修复梯度函数调用错误，对齐论文定义）
    参考：https://ieeexplore.ieee.org/document/5567109
    """
    # 输入校验
    if original.shape != denoised.shape:
        raise ValueError(f"图像尺寸不匹配: {original.shape} vs {denoised.shape}")
    
    # 转换为float32并归一化
    original = original.astype(np.float32) / data_range
    denoised = denoised.astype(np.float32) / data_range
    
    # 计算梯度（修复Sobel函数调用错误）
    def gradient(img):
        # 正确的Sobel参数：图像, 数据类型, x方向导数, y方向导数, 核大小
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)  # x方向梯度
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)  # y方向梯度
        return np.sqrt(gx**2 + gy**2 + 1e-8)  # 梯度幅值，避免除以零
    
    # 计算梯度幅值
    g_original = gradient(original)
    g_denoised = gradient(denoised)
    
    # 计算特征相似性分量
    numerator = 2 * g_original * g_denoised + 0.001
    denominator = g_original**2 + g_denoised**2 + 0.001
    fsim_map = numerator / denominator
    
    # 计算亮度相似性分量
    l_original = original
    l_denoised = denoised
    numerator_l = 2 * l_original * l_denoised + 0.01
    denominator_l = l_original**2 + l_denoised**2 + 0.01
    lsim_map = numerator_l / denominator_l
    
    # 综合FSIM并返回
    return np.mean(fsim_map * lsim_map)

def run_single_experiment(original_img, noise_density):
    """运行单组实验（生成噪声→检测→去噪→评估）"""
    # 1. 生成含噪图像
    noisy_img, real_mask = add_salt_pepper_noise(
        original_img, 
        noise_density=noise_density, 
        return_mask=True
    )
    
    # 2. 噪声检测
    noise_mask = noise_detection(noisy_img)
    
    # 3. 两种方法去噪
    baseline_result = adaptive_median_filter(noisy_img)  # 基线方法
    lsq_result = lsq_restore(noisy_img, noise_mask)      # 本文方法
    
    # 4. 计算评估指标
    metrics = {
        "noisy": {
            "psnr": psnr(original_img, noisy_img, data_range=255),
            "fsim": fsim(original_img, noisy_img) * 100  # 转为百分比
        },
        "baseline": {
            "psnr": psnr(original_img, baseline_result, data_range=255),
            "fsim": fsim(original_img, baseline_result) * 100
        },
        "lsq": {
            "psnr": psnr(original_img, lsq_result, data_range=255),
            "fsim": fsim(original_img, lsq_result) * 100
        }
    }
    
    return {
        "noisy_img": noisy_img,
        "baseline_img": baseline_result,
        "lsq_img": lsq_result,
        "metrics": metrics,
        "noise_density": noise_density
    }

def save_results(exp_result, save_dir, scene_name):
    """保存实验结果（图像和指标）"""
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    density = exp_result["noise_density"]
    
    # 保存图像
    cv2.imwrite(
        os.path.join(save_dir, "images", f"{scene_name}_noisy_{density:.1f}.png"),
        exp_result["noisy_img"]
    )
    cv2.imwrite(
        os.path.join(save_dir, "images", f"{scene_name}_baseline_{density:.1f}.png"),
        exp_result["baseline_img"]
    )
    cv2.imwrite(
        os.path.join(save_dir, "images", f"{scene_name}_lsq_{density:.1f}.png"),
        exp_result["lsq_img"]
    )
    
    # 保存指标
    with open(os.path.join(save_dir, f"{scene_name}_metrics.txt"), "a") as f:
        f.write(f"噪声密度: {density:.2f}\n")
        f.write(f"含噪图像 - PSNR: {exp_result['metrics']['noisy']['psnr']:.2f} dB, FSIM: {exp_result['metrics']['noisy']['fsim']:.2f}%\n")
        f.write(f"基线方法 - PSNR: {exp_result['metrics']['baseline']['psnr']:.2f} dB, FSIM: {exp_result['metrics']['baseline']['fsim']:.2f}%\n")
        f.write(f"LSQ方法  - PSNR: {exp_result['metrics']['lsq']['psnr']:.2f} dB, FSIM: {exp_result['metrics']['lsq']['fsim']:.2f}%\n")
        f.write("-"*50 + "\n")

def plot_comparison(original_img, exp_result, save_path):
    """绘制对比图并保存（确保中文显示正常）"""
    density = exp_result["noise_density"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("原始图像")
    axes[0].axis("off")
    
    # 含噪图像
    axes[1].imshow(exp_result["noisy_img"], cmap="gray")
    axes[1].set_title(f"含噪图像 (密度={density:.1f})\nPSNR: {exp_result['metrics']['noisy']['psnr']:.2f} dB")
    axes[1].axis("off")
    
    # 基线方法结果
    axes[2].imshow(exp_result["baseline_img"], cmap="gray")
    axes[2].set_title(f"自适应中值滤波\nPSNR: {exp_result['metrics']['baseline']['psnr']:.2f} dB")
    axes[2].axis("off")
    
    # LSQ方法结果
    axes[3].imshow(exp_result["lsq_img"], cmap="gray")
    axes[3].set_title(f"最小二乘恢复\nPSNR: {exp_result['metrics']['lsq']['psnr']:.2f} dB")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭图表释放资源

def main():
    # 实验配置
    scene_name = "scene1"
    original_path = "D:/LSQ_IR_Denoise/data/original/Data/scene1.png"  # 使用/避免转义问题
    save_root = "D:/LSQ_IR_Denoise/data/experiment_results"
    noise_densities = [0.2, 0.4, 0.6, 0.8]  # 测试不同噪声密度
    
    # 创建保存目录
    os.makedirs(save_root, exist_ok=True)
    
    try:
        # 读取原始图像
        original_img = read_ir_image(original_path)
        print(f"开始实验，原始图像尺寸: {original_img.shape}")
        
        # 遍历不同噪声密度
        for density in noise_densities:
            print(f"\n----- 噪声密度: {density} -----")
            exp_result = run_single_experiment(original_img, density)
            save_results(exp_result, save_root, scene_name)
            
            # 生成并保存对比图
            plot_path = os.path.join(save_root, f"{scene_name}_comparison_{density:.1f}.png")
            plot_comparison(original_img, exp_result, plot_path)
            print(f"对比图已保存至: {plot_path}")
            
            # 打印指标
            print(f"含噪图像: PSNR={exp_result['metrics']['noisy']['psnr']:.2f} dB, FSIM={exp_result['metrics']['noisy']['fsim']:.2f}%")
            print(f"基线方法: PSNR={exp_result['metrics']['baseline']['psnr']:.2f} dB, FSIM={exp_result['metrics']['baseline']['fsim']:.2f}%")
            print(f"LSQ方法:  PSNR={exp_result['metrics']['lsq']['psnr']:.2f} dB, FSIM={exp_result['metrics']['lsq']['fsim']:.2f}%")
        
        print("\n所有实验完成！结果已保存至:", save_root)
        
    except Exception as e:
        print(f"实验失败: {str(e)}")
        # 打印详细错误信息便于调试
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
# 运行实验脚本：生成含噪图像、检测噪声、去噪、评估并保存结果