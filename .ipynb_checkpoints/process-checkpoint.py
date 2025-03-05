import os
import cv2
import numpy as np

def enhance_purple_cells(image):
    """
    检测并增强紫色区域，同时保留其他区域不变。
    """
    # 定义紫色的颜色范围（BGR颜色空间）
    lower_purple = np.array([100, 0, 100])  # 较暗的紫色
    upper_purple = np.array([255, 100, 255])  # 较亮的紫色

    # 创建紫色掩码
    purple_mask = cv2.inRange(image, lower_purple, upper_purple)

    # 将图像转换到 HSV 空间，便于调整亮度和饱和度
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 在紫色区域上增加亮度和饱和度
    s_enhanced = np.copy(s)
    v_enhanced = np.copy(v)

    # 增加紫色区域的饱和度和亮度
    s_enhanced[purple_mask > 0] = np.clip(s[purple_mask > 0] + 50, 0, 255)
    v_enhanced[purple_mask > 0] = np.clip(v[purple_mask > 0] + 50, 0, 255)

    # 合并增强后的通道
    hsv_enhanced = cv2.merge((h, s_enhanced, v_enhanced))
    purple_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return purple_enhanced


def enhance_cell_edges(image_path, output_path):
    """
    对细胞图像进行紫色区域增强，并保存结果。
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告: 图像 {image_path} 未找到，跳过处理。")
        return

    # 2. 增强紫色区域
    purple_enhanced = enhance_purple_cells(image)

    # 3. 保存结果
    cv2.imwrite(output_path, purple_enhanced)
    print(f"处理完成: {image_path} -> {output_path}")


def process_dataset(input_root, output_root):
    """
    遍历数据集中的 train、val 和 test 文件夹，对每张图片进行处理并保存结果。
    """
    # 遍历 train、val、test 文件夹
    for split in ['train', 'val', 'test']:
        input_dir = os.path.join(input_root, split)
        output_dir = os.path.join(output_root, split)

        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)

        # 遍历输入文件夹中的所有图片
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # 仅处理 JPG 和 PNG 文件
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                # 对图片进行处理并保存结果
                enhance_cell_edges(input_path, output_path)


# 输入和输出路径
input_root = 'data-pre/imgs2'  # 输入数据集的根目录
output_root = 'data-pre/img3'  # 输出数据集的根目录

# 处理数据集
process_dataset(input_root, output_root)
print("所有图片处理完成！")