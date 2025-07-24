import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
import torch.nn.functional as F
from torchinfo import summary

class StdDevNorm(nn.Module):
    def __init__(self, input_channl, stddev_feat=1, stddev_group=4):
        super().__init__()
        self.stddev_feat = stddev_feat
        self.stddev_group = stddev_group
        self.conv = nn.Conv2d(input_channl + 1, input_channl, 1)
        
    def forward(self, input):
        batch, channel, height, width = input.shape # (B, C, H, W)
        group = min(batch, self.stddev_group)
        stddev = input.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width) # ->(B, 1, 1, C, H, W)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        output = torch.cat([input, stddev], 1)
        output = self.conv(output)
        return output

class SelfAttention(nn.Module):
    """Self Attention Layer"""
    def __init__(self, in_channels, activation='relu', k=8):
        super(SelfAttention, self).__init__()
        self.in_channels =  in_channels
        self.activation = activation
        
        self.W_query = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_key = nn.Conv2d(in_channels=in_channels, out_channels=(in_channels // k), kernel_size=1)
        self.W_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):
        """
        Input:
            X: (B, C, W, H)
        Output:
            output: (B, C, W, H) self attention value + input feature
            attention: (B, N, N)
        """
        B, C, W, H = X.size()
        
        queries = self.W_query(X).view(B, -1, W*H).permute(0, 2, 1) 
        # (B, C//k, W, H) -> (B, C//k, W*H) -> (B, W*H, C//k) = (B, N, C')
        
        keys = self.W_key(X).view(B, -1, W*H)
        # (B, C//k, W, H) -> (B, C//k, W*H) = (B, C', N)
        
        values = self.W_value(X).view(B, -1 ,W*H)
        # (B, C, W, H) -> (B, C, W*H) = (B, C, N)

        qk = torch.bmm(queries, keys)
        # (B, N, C')*(B, C', N) = (B, N, N)
        
        attention = self.softmax(qk)
        # (B, N, N)
        
        output = torch.bmm(values, attention.permute(0, 2, 1))
        # (B, C, N)*(B, N, N) = (B, C, N)
        
        output = output.view(B, C, W, H)
        # (B, C, N) -> (B, C, W, H)
        
        output = self.gamma * output + X
        # (B, C, W, H)
        
        return output, attention

class FeatureMatchDiscriminator(nn.Module):
    def __init__(self, img_size=256, conv_dim=64):
        super(FeatureMatchDiscriminator, self).__init__()
        self.img_size = img_size
        
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        # layer1
        layer1.append(SpectralNorm(nn.Conv2d(10, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim
        
        # layer2
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim * 2
        
        # layer3
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention1_dim = curr_dim
        
        # layer4
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2
        attention2_dim = curr_dim
        
        
        # last
        last.append(nn.Conv2d(curr_dim, 512, img_size // 16))
        
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attention1 = SelfAttention(attention1_dim)
        self.attention2 = SelfAttention(attention2_dim)
        
        # MLP Project Head
        self.MLP = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        
        # Upsample
        self.Upsample1 = nn.ConvTranspose2d(512, 512//2, 4, 2, 1)
        self.Upsample2 = nn.ConvTranspose2d(256, 256//2, 4, 2, 1)
        self.Upsample3 = nn.ConvTranspose2d(128, 128//2, 4, 2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.weights = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X) -> list:
        features = []
        
        output = self.layer1(X)
        feature1 = output
        features.append(feature1) # 0: Feature map layer: (B, 10, W, H) -> (B, 64, W//2, H//2)
        
        output = self.layer2(output)
        feature2 = output
        features.append(feature2) # 1: Feature map layer: -> (B, 128, W//4, H//4)
        
        output = self.layer3(output)
        output, p1 = self.attention1(output)
        feature3 = output
        features.append(feature3) # 2: Feature map layer: -> (B, 256, W//8, H//8)
        
        output = self.layer4(output)
        output, p2 = self.attention2(output)
        feature4 = output
        features.append(feature4) # 3: Feature map layer: -> (B, 512, W//16, H//16)
        
        output = self.last(output) # -> (B, 512, 1, 1)
        output = output.reshape(output.shape[0], -1) # -> (B, 512)
        features.append(output) # 4: Final output to calculate GAN loss: ->(B, 512)
        
        output = self.MLP(output) # -> (B, 64)
        features.append(output) # 5: Project output to calculate MMD loss: ->(B, 64)
        
        # Attention Module
        feature4_up = self.Upsample1(feature4) # (B, 512, W//16, H//16) -> (B, 256, W//8, H//8)
        feature3_combine = feature3 + feature4_up
        
        feature3_up = self.Upsample2(feature3_combine) # (B, 256, W//8, H//8) -> (B, 128, W//4, H//4)
        feature2_combine = feature2 + feature3_up
        
        feature2_up = self.Upsample3(feature2_combine) # (B, 128, W//4, H//4) -> (B, 64, W//2, H//2)
        feature1_combine = feature1 + feature2_up
        
        feature_avg = self.avg(feature1_combine).reshape(feature1_combine.shape[0], -1) # (B, 64, W//4, H//4) -> (B, 64)
        feature_attention = self.weights(feature_avg) # (B, 64) -> (B, 4)
        feature_attention = self.softmax(feature_attention) # (B, 4) -> Softmax
        
        for i in range(feature_attention.shape[1]):
            attention = feature_attention[:, i].reshape(feature_attention.shape[0], 1, 1, 1)
            features[i] = features[i] * attention
        
        return features

if __name__ == "__main__":
    D = FeatureMatchDiscriminator().cuda()
    X = torch.normal(0, 1, size=(8, 10, 256, 256)).cuda() # (B, C, W, H)
    Y = D(X)
    print(f"Input X: {X.shape} Output Y: {Y[-2].shape}")
    summary(D, X.shape, device="cuda")