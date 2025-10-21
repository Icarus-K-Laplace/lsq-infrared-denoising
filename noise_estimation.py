import numpy as np
import cv2
from skimage.morphology import square, opening, closing
from data_utils import read_ir_image, add_salt_pepper_noise

def get_neighborhood(img, x, y, win_size=3):
    """获取邻域（边缘复制填充，适配红外图像）"""
    h, w = img.shape
    pad = win_size // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    neighbor = img_padded[x:x+win_size, y:y+win_size]
    return neighbor

def noise_detection(img, win_size=3, T=20, sigma_thresh=1.0, candidate_thresh=15):
    """
    针对性修改的噪声检测（解决漏检/误检问题，适配椒盐噪声+红外图像）
    关键修改：放宽候选范围、降低偏差阈值、温和形态学处理
    """
    h, w = img.shape
    noise_mask = np.zeros((h, w), dtype=np.uint8)
    img_float = img.astype(np.float32)

    for x in range(h):
        for y in range(w):
            pixel = img_float[x, y]
            # -------------------------- 修改1：放宽候选噪声范围（解决漏检） --------------------------
            # 原逻辑：仅≤5或≥250为候选，可能漏检接近极值的噪声（如10、245）
            # 新逻辑：≤candidate_thresh（15）或≥(255-candidate_thresh)（240），覆盖更多真实椒盐噪声
            if not (pixel <= candidate_thresh or pixel >= (255 - candidate_thresh)):
                continue  # 仅保留接近极值的像素作为候选

            # -------------------------- 修改2：优化邻域分析（减少误判） --------------------------
            neighbor = get_neighborhood(img_float, x, y, win_size)
            # 非极值像素：排除候选范围内的像素（更合理区分正常/噪声）
            non_extreme = neighbor[(neighbor > candidate_thresh) & (neighbor < (255 - candidate_thresh))]
            extreme_low = (neighbor <= candidate_thresh).sum()    # 接近0的极值数量
            extreme_high = (neighbor >= (255 - candidate_thresh)).sum()  # 接近255的极值数量

            # -------------------------- 修改3：降低偏差判断阈值（解决漏检） --------------------------
            # 原逻辑：sigma_thresh=1.2，判断过严导致漏检；新逻辑=1.0，更易检测到真实噪声
            if len(non_extreme) > 0:
                mu = non_extreme.mean()
                sigma = non_extreme.std() if len(non_extreme) > 1 else 1e-6  # 避免除零
                
                # 分方向判断：低灰度候选→低于均值，高灰度候选→高于均值
                if pixel <= candidate_thresh:
                    if mu - pixel >= sigma_thresh * sigma:  # 降低阈值，更易标记为噪声
                        noise_mask[x, y] = 1
                else:
                    if pixel - mu >= sigma_thresh * sigma:
                        noise_mask[x, y] = 1
            else:
                # 无有效非极值像素时，放宽极值数量阈值（原T=15→新T=20），减少漏检
                if (pixel <= candidate_thresh and extreme_low <= T) or (pixel >= (255 - candidate_thresh) and extreme_high <= T):
                    noise_mask[x, y] = 1

    # -------------------------- 修改4：温和形态学处理（避免过度删除噪声标记） --------------------------
    # 原逻辑：square(2)结构元，可能删除小噪声块；新逻辑：square(1)，仅去除孤立1个像素的误检
    noise_mask = opening(noise_mask, square(1))  # 去除孤立误检点（1×1结构元）
    noise_mask = closing(noise_mask, square(1))  # 填补噪声块小空洞（1×1结构元）
    
    return noise_mask

# -------------------------- 新增：噪声检测评估函数（方便查看是否准确） --------------------------
def evaluate_noise_detection(real_mask, detected_mask):
    """
    评估噪声检测准确性（关键：召回率=不漏检，精确率=不误检）
    real_mask：真实噪声掩码（add_salt_pepper_noise返回的）
    detected_mask：检测到的噪声掩码
    """
    true_pos = np.sum((real_mask == 1) & (detected_mask == 1))  # 正确检测的噪声
    false_neg = np.sum((real_mask == 1) & (detected_mask == 0))  # 漏检的噪声（致命！）
    false_pos = np.sum((real_mask == 0) & (detected_mask == 1))  # 误检的正常像素
    
    recall = true_pos / (true_pos + false_neg + 1e-6)  # 召回率：越高越好（漏检少）
    precision = true_pos / (true_pos + false_pos + 1e-6)  # 精确率：越高越好（误检少）
    
    print(f"\n噪声检测评估（关键指标）：")
    print(f"召回率：{recall:.2%} → 越高越好（低于80%说明漏检太多，LSQ无法修复）")
    print(f"精确率：{precision:.2%} → 越高越好（低于70%说明误检太多，LSQ修坏正常像素）")
    return recall, precision

# 测试：运行噪声检测并评估（必须先确保检测准确）
if __name__ == "__main__":
    import os
    # 1. 生成已知噪声密度的含噪图（用设定0.4，验证检测准确性）
    original_path = "D:/LSQ_IR_Denoise/data/original/Data/scene1.png"
    original_img = read_ir_image(original_path)
    noisy_img, real_mask = add_salt_pepper_noise(original_img, noise_density=0.4, return_mask=True)

    # 2. 运行修改后的噪声检测
    detected_mask = noise_detection(noisy_img)

    # 3. 评估检测准确性（必须看召回率是否≥80%）
    recall, precision = evaluate_noise_detection(real_mask, detected_mask)

    # 4. 保存掩码可视化（查看检测效果）
    save_dir = "D:/LSQ_IR_Denoise/data/noisy/"
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "detected_mask_0.4.png"), detected_mask * 255)
    cv2.imwrite(os.path.join(save_dir, "real_mask_0.4.png"), real_mask * 255)
    print(f"\n掩码已保存：真实掩码(real_mask_0.4.png)、检测掩码(detected_mask_0.4.png)")