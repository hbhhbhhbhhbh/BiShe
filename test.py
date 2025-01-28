import cv2

# 加载图片
result = cv2.imread('output.jpg')

# 打印图片的尺寸
print(f"Image shape: {result.shape}")  # (height, width, channels)

# 打印前几个像素值
print(result[:5])  # 打印前5个像素的值
