import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

# 读取 PNG 图像 (RGB 图像)
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')  # 确保是 RGB 图像
    img = np.array(img)  # 转为 NumPy 数组
    return img

# 读取 JPG 图像 (黑白掩码)
def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')  # 转为灰度图
    mask = np.array(mask)  # 转为 NumPy 数组
    return mask
def save_image(image, output_path):
    """
    保存图像到文件
    :param image: 叠加后的图像
    :param output_path: 输出文件路径
    """
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 保存为 BGR 格式

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    将掩码叠加到图像上
    :param image: 原始图像 (H, W, C) RGB 图像
    :param mask: 黑白掩码 (H, W)，掩码为0表示背景，为1表示前景
    :param alpha: 掩码的透明度，取值范围为[0, 1]
    :return: 叠加后的图像
    """
    # 创建一个红色的掩码图像
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = [255, 0, 0]  # 红色

    # 将掩码和原图叠加，调整透明度
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    return overlay

# 假设 img_path 和 mask_path 是 PNG 和 JPG 文件路径
img_path = 'baso3-1.jpg'  # 修改为你的PNG文件路径
mask_path = 'output2.jpg'  # 修改为你的JPG掩码路径
output_path = 'UNet.png'  # 叠加图像保存路径

# 加载图像和掩码
image = load_image(img_path)  # [H, W, C] 图像
mask = load_mask(mask_path)   # [H, W] 掩码

# 叠加图像和掩码
overlay_image = overlay_mask_on_image(image, mask, alpha=0.5)

# 保存叠加后的图像
save_image(overlay_image, output_path)
# 使用 Matplotlib 显示叠加后的图像
plt.imshow(overlay_image)
plt.axis('off')
plt.show()
