import numpy as np
import math

import torch.nn.functional as F
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SpectralNorm
from torchinfo import summary

class PixelNorm(nn.Module):
    """像素归一化"""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)
    
def fused_leaky_relu(input, bias, negative_slope=0.2, scale= 2 ** 0.5):
    return scale * F.leaky_relu(
        input + bias.view((1, -1) + (1,) * (len(input.shape) - 2)), 
        negative_slope=negative_slope
                                )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0, activation=None  
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.activation = activation
        
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        
    def forward(self, input):
        if self.activation:
            # 平替
            output = F.linear(input, self.weight * self.scale)
            output = fused_leaky_relu(output, self.bias * self.lr_mul)
            # 自定义
            #output = F.linear(input, self.weight * self.scale, self.bias * self.lr_mul)
            #output = F.leaky_relu(output, 0.2)
            
        else:
            output = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

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

class FMG(nn.Module):
    def __init__(self, img_size=256, z_dim=512, conv_dim=64):
        super(FMG, self).__init__()
        
        self.img_size = img_size
        
        self.z_dim = z_dim
        
        # mlp = []
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []
        
        # if n_mlp:
        #     for i in range(n_mlp):
        #         mlp.append(EqualLinear(z_dim, z_dim, lr_mul=lr_mlp, activation='leaky_relu'))
            
        repeat_num = int(np.log2(self.img_size)) - 6 # =2
        multi = 2 ** repeat_num # =4
        
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * multi, 8, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * multi))
        layer1.append(nn.ReLU())
        
        curr_dim = conv_dim * multi
        
        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(curr_dim // 2))
        layer2.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        
        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(curr_dim // 2))
        layer3.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn1_dim = curr_dim
        
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(curr_dim // 2))
        layer4.append(nn.ReLU())
        
        curr_dim = curr_dim // 2
        attn2_dim = curr_dim
        
        last.append(nn.ConvTranspose2d(curr_dim, 10, 4, 2, 1))
        last.append(nn.Tanh())
        
        # self.mlp = nn.Sequential(*mlp)
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)
        
        self.attn1 = SelfAttention(attn1_dim)
        self.attn2 = SelfAttention(attn2_dim)
    
    def forward(self, z):
        feat_list = []
        output = self.layer1(z)
        feat_list.append(output) # 0 
        output = self.layer2(output)
        feat_list.append(output) # 1
        output = self.layer3(output)
        feat_list.append(output) # 2
        output, p1 = self.attn1(output)
        feat_list.append(output) # 3
        output = self.layer4(output)
        feat_list.append(output) # 4
        output, p2 = self.attn2(output)
        feat_list.append(output) # 5
        output = self.last(output)
        feat_list.append(output) # 6
         
        return output
    
if __name__ == "__main__":
    
    z = torch.randn(8, 512, 3, 3).cuda() # batchsize z_dim
    FMG = FMG().cuda()
    FMG = torch.nn.DataParallel(FMG)
    output = FMG(z)
    print(f"Input X: {z.shape} Output Y: {output.shape}")
    summary(FMG, z.shape, device="cuda")
