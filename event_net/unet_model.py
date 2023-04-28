""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_event(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_event, self).__init__()
        self.n_channels = n_channels # n_channels = 6 for image pairs
        self.n_classes = n_classes # n_classes = 2 for positive/negative
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits
        events = self.outc(x)
        return events

class UNet_2heads(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2, bilinear=True):
        super(UNet_2heads, self).__init__()
        self.n_channels = n_channels # n_channels = 6 for image pairs
        self.n_classes1 = n_classes1 # n_classes1 = 2 for positive/negative
        self.n_classes2 = n_classes2 # n_classes2 = 2 for events happening/no events happening
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # head 1 for event number prediction
        self.up1_1 = Up(1024, 512 // factor, bilinear)
        self.up2_1 = Up(512, 256 // factor, bilinear)
        self.up3_1 = Up(256, 128 // factor, bilinear)
        self.up4_1 = Up(128, 64, bilinear)
        self.outc_1 = OutConv(64, n_classes1)

        # head 2 for event existence prediction
        self.up1_2 = Up(1024, 512 // factor, bilinear)
        self.up2_2 = Up(512, 256 // factor, bilinear)
        self.up3_2 = Up(256, 128 // factor, bilinear)
        self.up4_2 = Up(128, 64, bilinear)
        self.outc_2 = OutConv(64, n_classes2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # head 1 for event number prediction
        x_1 = self.up1_1(x5, x4)
        x_1 = self.up2_1(x_1, x3)
        x_1 = self.up3_1(x_1, x2)
        x_1 = self.up4_1(x_1, x1)
        events = self.outc_1(x_1)

        # head 2 for event existence prediction
        x_2 = self.up1_2(x5, x4)
        x_2 = self.up2_2(x_2, x3)
        x_2 = self.up3_2(x_2, x2)
        x_2 = self.up4_2(x_2, x1)
        # logits = self.outc_2(x_2)
        logits = F.sigmoid(self.outc_2(x_2))
       
        return events, logits