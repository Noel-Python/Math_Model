import sys
sys.path.append("2023F/Dataloader")
from data_loader import get_loader

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
from torchinfo import summary

# 残差卷积
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
    
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)
        
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        )
        
        self.residual_conv_1 = ResidualConv(64, 128, 2, 1)
        self.residual_conv_2 = ResidualConv(128, 256, 2, 1)
        self.residual_conv_3 = ResidualConv(256, 512, 2, 1)
        self.residual_conv_4 = ResidualConv(512, 512, 2, 1)
        self.residual_conv_5 = ResidualConv(512, 512, 2, 1)
        self.residual_conv_6 = ResidualConv(512, 512, 2, 1)
        
        self.bridge = ResidualConv(512, 512, 2, 1)

        # self.b7 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )
        
    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        x5 = self.residual_conv_4(x4)
        x6 = self.residual_conv_5(x5)
        x7 = self.residual_conv_6(x6)
        x8 = self.bridge(x7)

        return [x1, x2, x3, x4, x5, x6, x7, x8]
    
class Decoder(nn.Module):
    def __init__(self, out_channels, add_i):
        super(Decoder, self).__init__()
        self.upsample_1 = Upsample(512*add_i, 512, 2, 2)
        self.up_residual_conv1 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_2 = Upsample(512, 512, 2, 2)
        self.up_residual_conv2 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_3 = Upsample(512, 512, 2, 2)
        self.up_residual_conv3 = ResidualConv(512*(1+add_i), 512, 1, 1)
        
        self.upsample_4 = Upsample(512, 512, 2, 2)
        self.up_residual_conv4 = ResidualConv(512*(1+add_i), 256, 1, 1)
        
        self.upsample_5 = Upsample(256, 256, 2, 2)
        self.up_residual_conv5 = ResidualConv(256*(1+add_i), 128, 1, 1)
        
        self.upsample_6 = Upsample(128, 128, 2, 2)
        self.up_residual_conv6 = ResidualConv(128*(1+add_i), 64, 1, 1)
        
        self.upsample_7 = Upsample(64, 64, 2, 2)
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(64*(1+add_i), out_channels, 1, 1)
        )
    
    def forward(self, encoder_output):
        x1, x2, x3, x4, x5, x6, x7, x8 = encoder_output
        
        u1 = self.upsample_1(x8)
        c1 = torch.cat([u1, x7], dim=1)
        
        u2 = self.up_residual_conv1(c1)
        u2 = self.upsample_2(u2)
        c2 = torch.cat([u2, x6], dim=1)
        
        u3 = self.up_residual_conv2(c2)
        u3 = self.upsample_3(u3)
        c3 = torch.cat([u3, x5], dim=1)
        
        u4 = self.up_residual_conv3(c3)
        u4 = self.upsample_4(u4)
        c4 = torch.cat([u4, x4], dim=1)
        
        u5 = self.up_residual_conv4(c4)
        u5 = self.upsample_5(u5)
        c5 = torch.cat([u5, x3], dim=1)
        
        u6 = self.up_residual_conv5(c5)
        u6 = self.upsample_6(u6)
        c6 = torch.cat([u6, x2], dim=1)
        
        u7 = self.up_residual_conv6(c6)
        u7 = self.upsample_7(u7)
        c7 = torch.cat([u7, x1], dim=1)
        
        output = self.output_layer(c7)
        return [c1, c2, c3, c4, c5, c6, c7, output]
        
if __name__ == "__main__":
    E = Encoder(10)
    x = torch.randn((4, 10, 256, 256))
    y = E(x)
    # print(f"Input X: {x.shape} Output Y: {y[-1].shape}")
    # summary(E, x.shape, device="cuda")
    
    feats_total = [torch.cat((y[i], y[i], y[i]), dim=1) for i in range(8)]
    D = Decoder(10, 3)
    y = D(feats_total)
    print(f"Input X: {feats_total[-1].shape} Output Y: {y[-1].shape}")
    summary(D, [i.shape for i in feats_total])
    
    