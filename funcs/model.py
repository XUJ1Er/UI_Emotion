# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2023/10/11 17:24
# Author     ：XuJ1E
# version    ：python 3.8
# File       : model.py
"""
import torch
import timm
import math
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.key = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, dim)
        )
        self.value = nn.Linear(dim, dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x.clone()
        prem = x.permute(0, 2, 3, 1)
        value = self.value(prem).reshape(b, -1, c)
        gap = self.gap(x).permute(0, 2, 3, 1)
        key = self.key(prem) * gap
        key = key.reshape(b, -1, c)
        attn = self.attn(x).permute(0, 2, 3, 1)
        attn = self.norm(self.act(attn)).reshape(b, -1, c)
        attn = self.softmax(attn + key) * value
        attn = attn.reshape(b, h, w, c)
        attn = attn.permute(0, 3, 1, 2)
        return attn + shortcut


class head(nn.Module):
    def __init__(self, dim=256, step=1):
        super().__init__()
        self.fc1 = nn.Linear(dim * step, dim * step)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * step, 7)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FERModel(nn.Module):
    def __init__(self, weight_dir='random', cuda=True):
        super().__init__()
        self.cuda = cuda
        self.net = timm.create_model('convnext_base', pretrained=False, num_classes=7, drop_path_rate=0.25)
        self.attention = Attention(dim=1024, kernel_size=7)
        self.step_head = torch.nn.ModuleList([head(step=2), head(step=3), head(step=4)])
        self._init_loads(weight_dir)

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.attention(x)
        x = x.mean([-2, -1])
        x0 = self.step_head[0](x[:, 0:512])
        x1 = self.step_head[1](x[:, 0:768])
        x2 = self.step_head[2](x[:, 0:1024])
        return x0+x1+x2

    def _init_loads(self, weight_dir='random'):
        if self.cuda:
            weights = torch.load(weight_dir)['state_dict']
        else:
            weights = torch.load(weight_dir, map_location=torch.device('cpu'))['state_dict']
        self.load_state_dict(weights, strict=False)


class vgg16face(nn.Module):
    mean = [0.5830, 0.4735, 0.4262]
    std = [0.2439, 0.1990, 0.1819]

    def __init__(self, weights_dir="random", cuda=True):
        super(vgg16face, self).__init__()

        self.cuda = cuda

        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_2_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_3_2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_3_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4_1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_4_2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_4_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_5_2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.conv_5_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7),
        )

        self._initialize_weights(weights_dir)

    def forward(self, x):
        # normalization
        x = F.normalize(x, self.mean, self.std)

        # forward
        x = self.conv_1_1(x)
        x = self.conv_1_2(x)
        x = self.pool_1(x)  # [bs, 64, 112, 112]

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.pool_2(x)  # [bs, 128, 56, 56]

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.pool_3(x)  # [bs, 256, 28, 28]

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.pool_4(x)  # [bs, 512, 14, 14]

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)
        x = self.pool_5(x)  # [bs, 512, 7, 7]

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, weights_dir="random"):
        if weights_dir == "random":  # random init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
        else:
            if self.cuda:
                weights = torch.load(weights_dir)
            else:
                weights = torch.load(weights_dir, map_location=torch.device('cpu'))
            self.load_state_dict(weights)
