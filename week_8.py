"""
编写YOLO v5 中的 C3, SPP, SPPF 等模块 实现分类
date: 2023-02-17
"""
import torch
from torch import nn
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, transforms
from torchvision.models import vgg16
import glob
import os, PIL, random, pathlib
from PIL import Image
import torchsummary as summary

total_dir = './weather_photos/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5])
])

total_data = torchvision.datasets.ImageFolder(total_dir, transform)
print(total_data)
print(total_data.class_to_idx)

idx_to_class = dict((v, k) for k,v in total_data.class_to_idx.items())
print(idx_to_class)

train_size = int(len(total_data) * 0.8)
test_size = int(len(total_data)) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def autopad(k, p=None):                        # kernel  padding 根据卷积核大小k自动计算卷积核padding数（0填充）
    """
    :param k: 卷积核的 kernel_size
    :param p: 卷积的padding  一般是None
    :return:  自动计算的需要pad值(0填充)
    """
    if p is None:
        # k 是 int 整数则除以2, 若干的整数值则循环整除
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, g=1):
        """
        :param c1: 输入的channel值
        :param c2: 输出的channel值
        :param k: 卷积的kernel_size
        :param s: 卷积的stride
        :param p: 卷积的padding  一般是None
        :param act: 激活函数类型   True就是SiLU(), False就是不使用激活函数
        :param g: 卷积的groups数  =1就是普通的卷积  >1就是深度可分离卷积
        """
        super(Conv, self).__init__()

        self.conv_1 = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.SiLU() if act else nn.Identity()     # 若act=True, 则激活,  act=False, 不激活

    def forward(self, x):

        return self.act(self.bn(self.conv_1(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, e=0.5, shortcut=True, g=1):
        """
        :param c1: 整个Bottleneck的输入channel
        :param c2: 整个Bottleneck的输出channel
        :param e: expansion ratio  c2*e 就是第一个卷积的输出channel=第二个卷积的输入channel
        :param shortcut: bool Bottleneck中是否有shortcut，默认True
        :param g: Bottleneck中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
        """
        super(Bottleneck, self).__init__()

        c_ = int(c2*e)                            # 使通道减半, c_具体多少取决于e
        self.conv_1 = Conv(c1, c_, 1, 1)
        self.conv_2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv_2(self.conv_1(x)) if self.add else self.conv_2(self.conv_1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        :param c1: 整个 C3 的输入channel
        :param c2: 整个 C3 的输出channel
        :param n: 有n个Bottleneck
        :param shortcut: bool Bottleneck中是否有shortcut，默认True
        :param g: C3中的3x3卷积类型  =1普通卷积  >1深度可分离卷积
        :param e: expansion ratio
        """
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv_1 = Conv(c1, c_, 1, 1)
        self.cv_2 = Conv(c1, c_, 1, 1)
        # *操作符可以把一个list拆开成一个个独立的元素,然后再送入Sequential来构造m,相当于m用了n次Bottleneck的操作
        self.m = nn.Sequential(*[Bottleneck(c_, c_, e=1, shortcut=True, g=1) for _ in range(n)])
        self.cv_3 = Conv(2*c_, c2, 1, 1)

    def forward(self, x):
        return self.cv_3(torch.cat((self.m(self.cv_1(x)), self.cv_2(x)), dim=1))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, e=0.5, n=1):
        """
        :param c1: 整个BottleneckCSP的输入channel
        :param c2: 整个BottleneckCSP的输出channel
        :param e: expansion ratio c2*e=中间其他所有层的卷积核个数/中间所有层的输入输出channel数
        :param n: 有 n 个Bottleneck
        """
        super(BottleneckCSP, self).__init__()
        c_ = int(c2*e)
        self.conv_1 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, e=1, shortcut=True, g=1) for _ in range(n)])
        self.conv_3 = Conv(c_, c_, 1, 1)
        self.conv_2 = Conv(c1, c_, 1, 1)
        self.bn = nn.BatchNorm2d(2*c_)
        self.LeakyRelu = nn.LeakyReLU()
        self.conv_4 = Conv(2*c_, c2, 1, 1)

    def forward(self, x):
        x_1 = self.conv_3(self.m(self.conv_1(x)))
        x_2 = self.conv_2(x)
        x_3 = torch.cat([x_1, x_2], dim=1)
        x_4 = self.LeakyRelu(self.bn(x_3))
        x = self.conv_4(x_4)
        return x


class SPP(nn.Module):
    def __init__(self, c1, c2, e=0.5, k1=5, k2=9, k3=13):
        """
        :param c1: SPP模块的输入channel
        :param c2: SPP模块的输出channel
        :param e: expansion ratio
        :param k1: Maxpool 的卷积核大小
        :param k2: Maxpool 的卷积核大小
        :param k3: Maxpool 的卷积核大小
        """
        super(SPP, self).__init__()
        c_ = int(c2*e)
        self.cv_1 = Conv(c1, c_, 1, 1)
        self.pool_1 = nn.MaxPool2d(kernel_size=k1, stride=1, padding=k1 // 2)
        self.pool_2 = nn.MaxPool2d(kernel_size=k2, stride=1, padding=k2 // 2)
        self.pool_3 = nn.MaxPool2d(kernel_size=k3, stride=1, padding=k3 // 2)
        self.cv_2 = Conv(4*c_, c2, 1, 1)

    def forward(self, x):
        return self.cv_2(torch.cat((self.pool_1(self.cv_1(x)), self.pool_2(self.cv_1(x)), self.pool_3(self.cv_1(x)), self.cv_1(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, e=0.5, k1=5):
        super(SPPF, self).__init__()
        c_ = int(c2*e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.pool_1 = nn.MaxPool2d(kernel_size=k1, stride=1, padding=k1 // 2)
        self.pool_2 = nn.MaxPool2d(kernel_size=k1, stride=1, padding=k1 // 2)
        self.pool_3 = nn.MaxPool2d(kernel_size=k1, stride=1, padding=k1 // 2)
        self.conv2 = Conv(4*c_, c2, 1, 1)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.pool_1(x_1)
        x_3 = self.pool_2(x_2)
        x_4 = self.pool_3(x_3)
        return self.conv2(torch.cat((x_1, x_2, x_3, x_4), dim=1))




class Concat(nn.Module):
    def __init__(self, dimension=1):
        """
        :param dimension: 沿着哪个维度进行concat
        """
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d(x))


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.conv = Conv(3, 32, 3, 2)                   # 3:输入通道  32:输出通道  3: kernel  2:stride
        self.spp = SPP(32, 64)
        self.c3 = C3(64, 128, n=1, shortcut=True, g=1, e=0.5)
        self.linear = nn.Sequential(
            nn.Linear(128*112*112, 1000),
            nn.ReLU(),

            nn.Linear(1000, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.spp(x)
        x = self.c3(x)
        x = x.view(-1, 128*112*112)
        x = self.linear(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model().to(device)

for x, y in train_dataloader:
    x, y = x.to(device), y.to(device)
    y_pre = model(x)
    break









