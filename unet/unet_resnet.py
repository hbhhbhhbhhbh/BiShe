""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .CBAM import *
from .resnet import resnet50

class UNetCBAMResnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetCBAMResnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        self.resnet=resnet50(False)
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.cbam = CBAM(512)
    def forward(self, x):
        # print("x:",x.shape)
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        [x1,x2, x3, x4, x5] = self.resnet.forward(x)
        # print("x1: ",x1.shape)
        # print("x2: ",x2.shape)
        # print("x3: ",x3.shape)
        # print("x4: ",x4.shape)
        # print("x5: ",x5.shape)
        # print("xx1: ",xx1.shape)
        # print("xx2: ",xx2.shape)
        # print("xx3: ",xx3.shape)
        # print("xx4: ",xx4.shape)
        # print("xx5: ",xx5.shape)
        x = self.up1(x5, x4)
        x = self.cbam (x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
        self.resnet=torch.utils.checkpoints(self.resnet)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)