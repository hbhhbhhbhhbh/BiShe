import cv2
import os

def enhance_contrast(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 分离亮度通道并进行直方图均衡化
    y, u, v = cv2.split(img_yuv)
    y_eq = cv2.equalizeHist(y)
    img_yuv_eq = cv2.merge((y_eq, u, v))

    # 转换回BGR色彩空间
    img_eq = cv2.cvtColor(img_yuv_eq, cv2.COLOR_YUV2BGR)
    return img_eq

# 处理图像并保存到新的目录
def process_images(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有子目录和文件
    for subdir in ['test', 'train', 'val']:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for filename in os.listdir(input_subdir):
            if filename.endswith(".jpg"):  # 检查是否为jpg文件
                image_path = os.path.join(input_subdir, filename)
                enhanced_img = enhance_contrast(image_path)

                # 保存增强后的图像到输出目录
                output_path = os.path.join(output_subdir, filename)
                cv2.imwrite(output_path, enhanced_img)

# 输入和输出目录
input_dir = './data-pre/imgs'
output_dir = './data-pre/imgs1'

# 处理图像
process_images(input_dir, output_dir)