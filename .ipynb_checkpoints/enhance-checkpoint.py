import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 数据集类
class CellEnhancementDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        """
        数据集类，加载图像和对应的掩码。
        :param img_dir: 包含图像的文件夹路径
        :param mask_dir: 包含掩码的文件夹路径
        :param img_transform: 图像的预处理操作
        :param mask_transform: 掩码的预处理操作
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 加载图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度图

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, images * masks)  # 只增强掩码区域
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, images * masks)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    print("Training complete!")

# 模型推理
def enhance_images(model, test_loader, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            enhanced_images = outputs * masks + images * (1 - masks)  # 只增强掩码区域

            # 保存增强后的图像
            save_image(enhanced_images, os.path.join(output_dir, f"enhanced_{idx}.png"))

# 主函数
if __name__ == "__main__":
    # 数据预处理
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整掩码大小
        transforms.ToTensor()
    ])

    # 数据集路径
    train_img_dir = "data-pre/imgs/train"
    train_mask_dir = "data-pre/masks/train"
    val_img_dir = "data-pre/imgs/val"
    val_mask_dir = "data-pre/masks/val"
    test_img_dir = "data-pre/imgs/test"
    test_mask_dir = "data-pre/masks/test"

    # 创建数据集
    train_dataset = CellEnhancementDataset(train_img_dir, train_mask_dir, img_transform=img_transform, mask_transform=mask_transform)
    val_dataset = CellEnhancementDataset(val_img_dir, val_mask_dir, img_transform=img_transform, mask_transform=mask_transform)
    test_dataset = CellEnhancementDataset(test_img_dir, test_mask_dir, img_transform=img_transform, mask_transform=mask_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 模型、损失函数和优化器
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # 输出目录
    output_dir = "data-pre/enhanced"
    enhance_images(model, test_loader, output_dir)

    print("Enhanced images saved to:", output_dir)