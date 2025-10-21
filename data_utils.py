import numpy as np
import cv2

def read_ir_image(img_path, gray=True):
    """读取红外图像（转为灰度图，适配0~255灰度范围）"""
    if gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取
    else:
        img = cv2.imread(img_path)
    
    # 检查图像是否读取成功
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}\n请检查路径或文件完整性")
    
    return img.astype(np.uint8)

def add_salt_pepper_noise(img, noise_density=0.2, return_mask=False):
    """
    添加椒盐噪声（支持返回真实噪声掩码，解决return_mask参数错误）
    参数：
        img: 原始图像（uint8）
        noise_density: 目标噪声密度（0~1）
        return_mask: 是否返回真实噪声掩码（True/False）
    返回：
        若return_mask=True：(noisy_img, real_mask)
        若return_mask=False：noisy_img
    """
    img = img.astype(np.uint8)
    h, w = img.shape
    noisy_img = img.copy()
    
    # 1. 分配椒、盐噪声概率（各占一半，总密度=noise_density）
    p_pepper = noise_density / 2  # 椒噪声（灰度0）概率
    p_salt = noise_density / 2    # 盐噪声（灰度255）概率
    
    # 2. 生成独立噪声掩码（避免同一像素被两种噪声覆盖）
    mask_pepper = np.random.choice([0, 1], size=(h, w), p=[1-p_pepper, p_pepper])
    mask_salt = np.random.choice([0, 1], size=(h, w), p=[1-p_salt, p_salt])
    mask_pepper[mask_salt == 1] = 0  # 盐噪声位置，椒噪声掩码置0
    
    # 3. 应用噪声到图像
    noisy_img[mask_pepper == 1] = 0
    noisy_img[mask_salt == 1] = 255
    
    # 4. 合并真实噪声掩码（1=噪声，0=正常）
    real_mask = mask_pepper | mask_salt
    
    # 5. 输出实际噪声密度（便于验证）
    real_density = np.sum(real_mask) / (h * w)
    print(f"目标噪声密度：{noise_density:.2f}，实际噪声密度：{real_density:.4f}")
    
    # 6. 根据参数返回结果
    if return_mask:
        return noisy_img, real_mask
    else:
        return noisy_img