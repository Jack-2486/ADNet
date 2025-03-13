import torch
import torch.nn as nn


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SENet(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SENet, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入的形状
        y = self.gap(x).view(b, c)  # 通过全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层
        return x * y.expand_as(x)  # 重新加权特征图
