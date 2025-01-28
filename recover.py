import cv2
import numpy as np

# 读取图像（确保是灰度图像）
image = cv2.imread('baso3-1.png', cv2.IMREAD_GRAYSCALE)

# 确保像素值是0和1
image = np.where(image > 0, 1, 0)

# 通过 OpenCV 阈值操作将 0 和 1 转换为黑白
_, binary_image = cv2.threshold(image.astype(np.uint8) * 255, 127, 255, cv2.THRESH_BINARY)

# 保存二值化后的图像
cv2.imwrite('binary_image.jpg', binary_image)
