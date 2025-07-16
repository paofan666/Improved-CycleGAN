import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)  # Q
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)  # K
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)  # V
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(B, -1, H * W)  # (B, C//8, H*W)
        energy = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = self.softmax(energy)  # (B, H*W, H*W)

        value = self.value(x).view(B, -1, H * W)  # (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        return out + x  # 残差连接


# 可变形卷积注意力机制
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DeformableSelfAttention(nn.Module):
    """Self-Attention with Deformable Convolution"""

    def __init__(self, in_channels):
        super(DeformableSelfAttention, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)  # 3x3 deformable offsets
        self.query = DeformConv2d(in_channels, in_channels // 8, kernel_size=3, padding=1)
        self.key = DeformConv2d(in_channels, in_channels // 8, kernel_size=3, padding=1)
        self.value = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放因子

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset_conv(x)  # 计算 DCN 偏移量

        # 计算 Query、Key、Value
        query = self.query(x, offset).view(B, H * W, -1)  # (B, HW, C')
        key = self.key(x, offset).view(B, H * W, -1).permute(0, 2, 1)  # (B, C', HW)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, HW, HW)

        value = self.value(x, offset).view(B, H * W, -1)  # (B, HW, C)
        out = torch.bmm(attention, value).view(B, C, H, W)  # (B, HW, HW) * (B, HW, C) -> (B, HW, C)

        return self.gamma * out + x  # 残差连接