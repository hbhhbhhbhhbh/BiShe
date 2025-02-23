import torch
import torch.nn as nn

class EdgeDetectionModule(nn.Module):
    def __init__(self):
        super(EdgeDetectionModule, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y.weight.data = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, x):
        # 确保输入张量的通道数为 1
        if x.shape[1] > 1:
            x = x[:, 0:1, :, :]  # 选择第一个通道
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_map = torch.clamp(edge_map, min=0)  # 将值限制在 [0, 1] 范围内
        edge_map = edge_map / (edge_map.max() + 1e-8)
        return edge_map