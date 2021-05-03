import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 96)
        self.dconv_down3 = double_conv(96, 128)
        self.dconv_down4 = double_conv(128, 164)
        self.dconv_down5 = double_conv(164,196)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = double_conv(164+164,164)
        self.dconv_up3 = double_conv(128 + 128, 128)
        self.dconv_up2 = double_conv(96 + 96, 96)
        self.dconv_up1 = double_conv(64 + 64, 64)

        self.out_1 = nn.Conv2d(64, n_class, 1)

        self.cat_1 = nn.Conv2d(96,64,1)
        self.cat_2 = nn.Conv2d(128,96,1)
        self.cat_3 = nn.Conv2d(164,128,1)
        self.cat_4 = nn.Conv2d(196,164,1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        x = self.dconv_down5(x)
        
        
        x = self.upsample(x)
        x = torch.cat([self.cat_4(x),conv4], dim=1)
        
        
        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([self.cat_3(x),conv3],dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([self.cat_2(x), conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([self.cat_1(x), conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        y1 = self.out_1(x)
        
        return y1
