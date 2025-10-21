import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def fsim(original, denoised, data_range=255):
    """
    优化的FSIM计算（修复梯度方向参数错误）
    参考：https://ieeexplore.ieee.org/document/5567109
    """
    # 输入校验：确保图像尺寸一致
    if original.shape != denoised.shape:
        raise ValueError(f"原始图与去噪图尺寸不匹配: {original.shape} vs {denoised.shape}")
    
    # 转换为float32并归一化，避免精度损失
    original = original.astype(np.float32) / data_range
    denoised = denoised.astype(np.float32) / data_range
    h, w = original.shape

    # 1. 计算相位一致性（修复梯度方向参数错误）
    def phase_consistency(img, sigma=1.0):
        # 高斯模糊减少噪声干扰
        img_blur = cv2.GaussianBlur(img, (3, 3), sigma)
        # 多方向梯度（0°, 45°, 90°, 135°）- 修正参数错误
        kernels = [
            cv2.getDerivKernels(1, 0, 3)[0],  # 水平 (dx=1, dy=0)
            cv2.getDerivKernels(1, 1, 3)[0],  # 对角线 (dx=1, dy=1)
            cv2.getDerivKernels(0, 1, 3)[0],  # 垂直 (dx=0, dy=1)
            cv2.getDerivKernels(1, 1, 3)[0].T  # 反对角线 (通过转置对角线核实现)
        ]
        # 计算各方向梯度幅值
        grads = [cv2.filter2D(img_blur, -1, k) for k in kernels]
        grad_mags = [np.abs(g) for g in grads]
        # 相位一致性 = 梯度总和 / (平均梯度 + 1e-8)
        sum_mag = np.sum(grad_mags, axis=0)
        mean_mag = np.mean(sum_mag) + 1e-8
        return sum_mag / mean_mag

    # 2. 计算特征相似性分量
    pc_original = phase_consistency(original)
    pc_denoised = phase_consistency(denoised)
    fsim_pc = (2 * pc_original * pc_denoised + 0.001) / (pc_original**2 + pc_denoised**2 + 0.001)

    # 3. 计算亮度相似性分量
    lsim = (2 * original * denoised + 0.01) / (original**2 + denoised**2 + 0.01)

    # 4. 空间加权（中心像素权重更高）
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w//2, h//2
    dist = np.sqrt((x - cx)** 2 + (y - cy)** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    weight = 1 - (dist / max_dist)  # 距离中心越近，权重越高

    # 5. 加权平均得到最终FSIM
    fsim_val = np.sum(fsim_pc * lsim * weight) / np.sum(weight)
    return fsim_val

def calculate_metrics(original_img, denoised_img):
    """计算PSNR和FSIM指标（增加异常处理和结果校验）"""
    # 输入类型校验
    if original_img.dtype != np.uint8 or denoised_img.dtype != np.uint8:
        raise TypeError("输入图像必须为uint8类型")
    
    # 转换为float32计算（避免整数溢出）
    original = original_img.astype(np.float32)
    denoised = denoised_img.astype(np.float32)

    # 计算PSNR
    psnr_val = psnr(original, denoised, data_range=255)

    # 计算FSIM并转为百分比
    fsim_val = fsim(original_img, denoised_img) * 100

    # 结果合理性校验
    if not (0 <= psnr_val <= 50):
        print(f"警告：PSNR值({psnr_val:.2f}dB)超出合理范围，可能图像输入错误")
    if not (0 <= fsim_val <= 100):
        print(f"警告：FSIM值({fsim_val:.2f}%)超出合理范围，可能计算错误")

    return round(psnr_val, 2), round(fsim_val, 2)

# 测试：验证指标计算准确性
if __name__ == "__main__":
    from data_utils import read_ir_image
    import os

    # 配置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_path = os.path.join(project_root, "data", "original", "Data", "scene1.png")
    denoised_path = os.path.join(project_root, "data", "denoised", "scene1_denoised_lsq.png")

    try:
        # 读取图像
        original_img = read_ir_image(original_path)
        denoised_img = read_ir_image(denoised_path)
        
        # 计算指标
        psnr_val, fsim_val = calculate_metrics(original_img, denoised_img)
        
        # 输出结果
        print(f"评估指标：")
        print(f"PSNR: {psnr_val} dB")
        print(f"FSIM: {fsim_val} %")
        print("\n论文对比参考：")
        print("若噪声密度0.8时，PSNR>22dB且FSIM>90%，结果基本合理")
        print("若LSQ方法比自适应中值滤波高1.5dB以上，符合论文预期")

    except Exception as e:
        print(f"计算失败：{str(e)}")