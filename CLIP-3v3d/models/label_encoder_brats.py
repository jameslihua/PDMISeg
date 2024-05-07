import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import math
'''
继承系统的3D卷积，然后多了一个权重归一化
'''
class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        # 求各维度均值 (每一维中，每一列的平均值)
        # weight_mean = weight.mean(dim=1, keepdim=True)
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)

        # 用权重减去权重均值
        weight = weight - weight_mean
        # 求标准差
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        # 用权重除以标准差
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), dilation=(1,1,1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)

# 应用在conresnet中，在上采样过程中进行上下文残差映射

# 应用在Conresnet中的layer层中，解决跳跃连接尺寸不一样的问题。
class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.LeakyReLU(inplace=True)

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class conresnet(nn.Module):
    def __init__(self, shape, block, layers, num_classes=2, weight_std=False):
        self.shape = shape
        self.weight_std = weight_std
        super(conresnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1_32 = nn.Sequential(
            conv3x3x3(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))
        self.conv_256_512 = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))
        
        self.conv_512_1024 = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(512, 1024, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))
        
        
        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        # 使用空洞卷积后特征图也会发生改变
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2,2,2))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    # block指的是 nobottleneck blocks 指的是 layer[1,2,2,2,2]
    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.LeakyReLU(inplace=True),  # inplace:是否将得到的值计算得到的值覆盖之前的值
                conv3x3x3(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )
        layers = []

        # 解决空洞卷积中，kernel不连续,未能使用到所有的pixel的问题
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1

        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))

        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x 1 2 128 128 128
        ## encoder
        #print(x.shape)
        c, _, _, _ = x.shape
        x = x.view(c, 1, 128, 128, 128)
        x = self.conv_1_32(x)
        x = self.layer0(x)			# [2,32,128,128,128]
        skip1 = x       

        x = self.conv_32_64(x)
        x = self.layer1(x)			# [2,64,64,64,64]
        skip2 = x       

        x = self.conv_64_128(x)
        x = self.layer2(x)			# [2,128,32,32,32]
        skip3 = x       

        x = self.conv_128_256(x)	# [2,256,16,16,16]
        x = self.layer3(x)  		# [2,256,16,16,16]
        # 
        x = self.layer4(x)   		# [2,256,16,16,16]
        x = self.conv_256_512(x)
        x = self.avg_pool(x)       #2 512 1 1 1
        x = x.view(c, 512)
        
        x = F.sigmoid(x)
        return x


