import numpy as np
import cv2
from scipy.optimize import leastsq
from noise_estimation import get_neighborhood  # 关联优化后的邻域函数
from data_utils import read_ir_image  # 用于测试

def cubic_fit_func(params, x):
    """三阶多项式拟合函数（严格对齐论文1.3节：y = a*x³ + b*x² + c*x + d）"""
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d

def fit_error(params, x, y):
    """最小二乘优化目标：残差=拟合值-真实值（确保优化方向正确）"""
    return cubic_fit_func(params, x) - y

def lsq_restore(noisy_img, noise_mask, min_valid_pixels=5, win_sizes=[3,5,7], max_fit_error=20):
    """
    最小二乘拟合恢复（针对性优化：解决漏修、拟合失真、修复值跳变）
    参数说明：
        noisy_img: 含噪图像（uint8，0~255）
        noise_mask: 噪声掩码（1=噪声，0=正常）
        min_valid_pixels: 拟合所需最小非噪声像素数（论文取5，避免拟合样本不足）
        win_sizes: 邻域递进顺序（3→5→7，优先小窗口保细节）
        max_fit_error: 拟合残差阈值（超过则用中值，避免拟合失真）
    返回：
        denoised_img: 去噪图像（uint8，0~255）
    """
    h, w = noisy_img.shape
    denoised_img = noisy_img.copy().astype(np.float32)  # 用float32保留拟合精度
    noise_mask = noise_mask.astype(np.uint8)  # 确保掩码格式统一

    # 遍历每个像素（仅处理噪声点）
    for x in range(h):
        for y in range(w):
            # 非噪声点直接保留，不处理
            if noise_mask[x, y] != 1:
                continue

            # -------------------------- 步骤1：递进扩大邻域，找足够非噪声像素（解决漏修） --------------------------
            valid_pixels = None  # 用于拟合的非噪声像素
            neighbor_size = None  # 最终选定的邻域大小
            # 按3→5→7递进扩大邻域，确保找到足够非噪声像素
            for win_size in win_sizes:
                # 获取当前邻域的灰度值和掩码
                neighbor_gray = get_neighborhood(denoised_img, x, y, win_size)
                neighbor_mask = get_neighborhood(noise_mask, x, y, win_size)
                # 提取邻域内的非噪声像素（仅用真实正常像素拟合，避免噪声干扰）
                current_valid = neighbor_gray[neighbor_mask == 0]
                
                # 若非噪声像素数满足要求，停止扩大邻域（小窗口保细节）
                if len(current_valid) >= min_valid_pixels:
                    valid_pixels = current_valid
                    neighbor_size = win_size
                    break

            # -------------------------- 步骤2：极端情况处理（避免拟合样本不足导致失真） --------------------------
            # 若所有邻域都不足5个非噪声像素，改用中值替代（论文隐含容错逻辑）
            if valid_pixels is None or len(valid_pixels) < min_valid_pixels:
                # 用3×3邻域中值（最小窗口，减少对周围影响）
                fallback_neighbor = get_neighborhood(denoised_img, x, y, 3)
                denoised_img[x, y] = np.median(fallback_neighbor)
                continue

            # -------------------------- 步骤3：三阶多项式拟合（提升稳定性，避免失真） --------------------------
            # 1. 数据预处理：排序+去异常值（避免极端值影响拟合）
            valid_pixels_sorted = np.sort(valid_pixels)
            # 去除首尾10%的异常值（如邻域内少量未标记的噪声）
            trim_ratio = 0.1
            trim_len = int(len(valid_pixels_sorted) * trim_ratio)
            if trim_len > 0:
                valid_pixels_trimmed = valid_pixels_sorted[trim_len:-trim_len]
            else:
                valid_pixels_trimmed = valid_pixels_sorted

            # 2. 拟合自变量/因变量（论文要求：按灰度排序后拟合）
            n = len(valid_pixels_trimmed)
            x_fit = np.arange(n)  # 自变量：像素排序后的序号（0~n-1）
            y_fit = valid_pixels_trimmed  # 因变量：排序后的非噪声像素灰度

            # 3. 优化初始参数（避免拟合发散，解决“局部最优”问题）
            # d初值=非噪声像素均值（贴近真实灰度范围），a/b/c初值=1e-6（极小值，避免初始偏差过大）
            mean_gray = np.mean(y_fit)
            params_init = [1e-6, 1e-6, 1e-6, mean_gray]

            # 4. 最小二乘拟合（增加迭代次数，确保收敛）
            try:
                params_opt, cov_params, infodict, mesg, ier = leastsq(
                    fit_error,
                    params_init,
                    args=(x_fit, y_fit),
                    maxfev=2000,  # 增加最大迭代次数，复杂邻域也能收敛
                    full_output=True  # 输出拟合信息，用于判断有效性
                )
                # 计算拟合残差（判断拟合是否有效）
                residuals = infodict['fvec']
                mean_residual = np.mean(np.abs(residuals))
            except Exception as e:
                # 拟合失败时，用非噪声像素的中值替代
                denoised_img[x, y] = np.median(valid_pixels)
                continue

            # -------------------------- 步骤4：修复值计算+约束（避免跳变，提升视觉效果） --------------------------
            # 1. 若拟合残差过大（拟合失真），改用非噪声像素中值
            if mean_residual > max_fit_error:
                restored_val = np.median(valid_pixels)
            else:
                # 论文核心：取拟合值的中值（抗拟合异常值）
                fit_values = cubic_fit_func(params_opt, x_fit)
                restored_val = np.median(fit_values)

            # 2. 灰度约束1：修复值在“非噪声像素均值±2倍标准差”内（避免跳变）
            std_gray = np.std(valid_pixels)
            restored_val = np.clip(restored_val, mean_gray - 2*std_gray, mean_gray + 2*std_gray)
            # 3. 灰度约束2：最终截断到0~255（符合图像格式）
            restored_val = np.clip(restored_val, 0, 255)

            # 赋值给去噪图像（四舍五入为整数灰度）
            denoised_img[x, y] = int(round(restored_val))

    # 转换回uint8格式，确保与输入图像兼容
    return denoised_img.astype(np.uint8)

# -------------------------- 测试：验证LSQ修复效果（必须先确保噪声检测准确） --------------------------
if __name__ == "__main__":
    import os
    from noise_estimation import noise_detection
    from src_metrics import calculate_metrics  # 用于计算指标验证

    # 1. 配置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_path = os.path.join(project_root, "data", "original", "Data", "scene1.png")
    noisy_path = os.path.join(project_root, "data", "noisy", "scene1_noisy_0.8.png")  # 0.8噪声密度的含噪图
    save_path = os.path.join(project_root, "data", "denoised", "scene1_lsq_test_0.8.png")

    # 2. 读取数据
    original_img = read_ir_image(original_path)
    noisy_img = read_ir_image(noisy_path)
    # 先运行优化后的噪声检测，获取准确掩码
    noise_mask = noise_detection(noisy_img)

    # 3. 运行LSQ修复
    denoised_img = lsq_restore(noisy_img, noise_mask)

    # 4. 计算指标（验证是否优于基线）
    psnr_noisy, fsim_noisy = calculate_metrics(original_img, noisy_img)
    psnr_lsq, fsim_lsq = calculate_metrics(original_img, denoised_img)

    # 5. 输出结果（关键看LSQ是否比含噪图提升明显）
    print(f"=== 0.4噪声密度下LSQ修复效果 ===")
    print(f"含噪图：PSNR={psnr_noisy} dB, FSIM={fsim_noisy}%")
    print(f"LSQ去噪图：PSNR={psnr_lsq} dB, FSIM={fsim_lsq}%")
    print(f"PSNR提升：{psnr_lsq - psnr_noisy:.2f} dB, FSIM提升：{fsim_lsq - fsim_noisy:.2f}%")

    # 6. 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, denoised_img)
    print(f"\nLSQ去噪图已保存：{save_path}")