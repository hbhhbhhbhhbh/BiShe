import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .unet_parts import *
from .CBAM import *
from .resnet import resnet50
from .edge_detection import EdgeDetectionModule
class DualBranchUNetCBAMResnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DualBranchUNetCBAMResnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name="DBUC"
        factor = 2 if bilinear else 1

        # # 使用 ResNet50 作为特征提取器
        # self.resnet = resnet50(pretrained=False)
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        
        # 主体分割分支
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 边缘分割分支
        self.edge_up1 = Up(1024, 512 // factor, bilinear)
        self.edge_up2 = Up(512, 256 // factor, bilinear)
        self.edge_up3 = Up(256, 128 // factor, bilinear)
        self.edge_up4 = Up(128, 64, bilinear)
        self.edge_outc = OutConv(64, n_classes)

        # CBAM 注意力机制
        self.cbam = CBAM(512)

        # 边缘检测模块
        self.edge_detector = EdgeDetectionModule()

    def forward(self, x):
        # [x1, x2, x3, x4, x5] = self.resnet.forward(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 主体分割分支
        x = self.up1(x5, x4)
        x = self.cbam(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # 边缘分割分支
        edge_x = self.edge_up1(x5, x4)
        edge_x = self.cbam(edge_x)
        edge_x = self.edge_up2(edge_x, x3)
        edge_x = self.edge_up3(edge_x, x2)
        edge_x = self.edge_up4(edge_x, x1)
        edge_logits = self.edge_outc(edge_x)

        # 特征融合
        edge_features = self.edge_detector(edge_logits)
        fused_logits = logits+edge_features
        # print("logits: ",logits.shape)
        return fused_logits,edge_logits

    def use_checkpointing(self):
        # self.resnet = torch.utils.checkpoint(self.resnet)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.edge_up1 = torch.utils.checkpoint(self.edge_up1)
        self.edge_up2 = torch.utils.checkpoint(self.edge_up2)
        self.edge_up3 = torch.utils.checkpoint(self.edge_up3)
        self.edge_up4 = torch.utils.checkpoint(self.edge_up4)
        self.edge_outc = torch.utils.checkpoint(self.edge_outc)