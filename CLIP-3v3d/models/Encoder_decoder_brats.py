import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import math

'''
继承系统的3D卷积，然后多了一个权重归一化
'''


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        # 求各维度均值 (每一维中，每一列的平均值)
        # weight_mean = weight.mean(dim=1, keepdim=True)
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)

        # 用权重减去权重均值
        weight = weight - weight_mean
        # 求标准差
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        # 用权重除以标准差
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
              bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


# 应用在conresnet中，在上采样过程中进行上下文残差映射
class ConResAtt(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(ConResAtt, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer

        self.relu = nn.LeakyReLU(inplace=True)
        # inplace = true :对从上层卷积网络中传递下来的tensor直接进行覆盖，这样能够节省运算内存，不用多存储其他变量

        self.gn_seg = nn.GroupNorm(8, in_planes)
        self.conv_seg = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                  stride=(stride[0], stride[1], stride[2]),
                                  padding=(padding[0], padding[1], padding[2]),
                                  dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                  weight_std=self.weight_std)

        self.gn_res = nn.GroupNorm(8, out_planes)
        self.conv_res = conv3x3x3(out_planes, out_planes, kernel_size=(1, 1, 1),
                                  stride=(1, 1, 1), padding=(0, 0, 0),
                                  dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                  weight_std=self.weight_std)

        self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.conv_res1 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                   stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                   dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                   weight_std=self.weight_std)

        self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.conv_res2 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                   stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                   dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                   weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_planes)
        self.conv_mp_first = conv3x3x3(4, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                       stride=(stride[0], stride[1], stride[2]),
                                       padding=(padding[0], padding[1], padding[2]),
                                       dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                       weight_std=self.weight_std)

        self.conv_mp = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                 stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                                 dilation=(dilation[0], dilation[1], dilation[2]), bias=bias,
                                 weight_std=self.weight_std)

    # # # 上下文残差G，每对相邻特征之间的位置方向绝对差。
    def _res(self, x):  # bs, channel, D, W, H

        bs, channel, depth, height, width = x.shape
        x_copy_d = torch.zeros_like(x).cuda()
        x_copy_h = torch.zeros_like(x).cuda()
        x_copy_w = torch.zeros_like(x).cuda()

        x_copy_d[:, :, 3:, :, :] = x[:, :, 0: depth - 3, :, :]
        x_copy_h[:, :, :, 1:, :] = x[:, :, :, 0: height - 1, :]
        x_copy_w[:, :, :, :, 1:] = x[:, :, :, :, 0: width - 1]

        res_d = x - x_copy_d
        res_h = x - x_copy_h
        res_w = x - x_copy_w

        res_d[:, :, 0:3, :, :] = 0
        res_h[:, :, :, 0, :] = 0
        res_w[:, :, :, :, 0] = 0

        res_d = torch.abs(res_d)
        res_h = torch.abs(res_h)
        res_w = torch.abs(res_w)

        return [res_d, res_h, res_w]

    def forward(self, input):
        x1, x2_d, x2_h, x2_w = input  # [seg_x4, res_x4] seg_X4(F)传入后变为X1
        if self.first_layer:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)  # 求出分割特征

            res = torch.sigmoid(x1)  # 将分割的结果传入残差层
            res = self._res(res)  # 计算残差
            res_d = res[0]
            res_h = res[1]
            res_w = res[2]

            res_d = self.conv_res(res_d)  # 残差卷积Conv(G) 求出残差特征
            res_h = self.conv_res(res_h)
            res_w = self.conv_res(res_w)

            x2_d = self.conv_mp_first(x2_d)  # 对低层特征进行卷积，使其shape相同
            x2_h = self.conv_mp_first(x2_h)
            x2_w = self.conv_mp_first(x2_w)
            # 将生成的上下文残差特征图与前一上下文残差层的残差特征进行融合 Conv(G)与Ires元素求和
            x2_d = x2_d + res_d
            x2_h = x2_h + res_h
            x2_w = x2_w + res_w

        else:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)  # 计算残差
            res_d = res[0]
            res_h = res[1]
            res_w = res[2]

            res_d = self.conv_res(res_d)  # 残差卷积Conv(G) 求出残差特征
            res_h = self.conv_res(res_h)
            res_w = self.conv_res(res_w)

            if self.in_planes != self.out_planes:
                x2_d = self.gn_mp(x2_d)
                x2_d = self.relu(x2_d)
                x2_d = self.conv_mp(x2_d)

                x2_h = self.gn_mp(x2_h)
                x2_h = self.relu(x2_h)
                x2_h = self.conv_mp(x2_h)

                x2_w = self.gn_mp(x2_w)
                x2_w = self.relu(x2_w)
                x2_w = self.conv_mp(x2_w)

            x2_d = x2_d + res_d
            x2_h = x2_h + res_h
            x2_w = x2_w + res_w

        # 使用加权层，对Conv(G)与Ires进行细化，得出Ores
        x2_d = self.gn_res1(x2_d)
        x2_d = self.relu(x2_d)
        x2_d = self.conv_res1(x2_d)

        x2_h = self.gn_res1(x2_h)
        x2_h = self.relu(x2_h)
        x2_h = self.conv_res1(x2_h)

        x2_w = self.gn_res1(x2_w)
        x2_w = self.relu(x2_w)
        x2_w = self.conv_res1(x2_w)

        x1 = x1 * (1 + torch.sigmoid(x2_d) + torch.sigmoid(x2_h) + torch.sigmoid(
            x2_w))  # 分割路径的上下文注意力加权输出 Oseg = F * (1 + sigmoid(Ores))

        return [x1, x2_d, x2_h, x2_w]

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # 共享权重的MLP
        self.fc1   = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
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
		#self.ca = ChannelAttention(planes)
        #self.sa = SpatialAttention()
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
		
        #seg = self.ca(seg) * seg
        #seg = self.sa(seg) * seg
        
        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class conresnet(nn.Module):
    def __init__(self, shape, block, layers, weight_std=False):
        self.shape = shape
        self.weight_std = weight_std
        super(conresnet, self).__init__()

        self.conv_1_32 = nn.Sequential(
            conv3x3x3(4, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

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

        self.conv_256_128 = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_64 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_32 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(inplace=True),
            conv3x3x3(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        # 使用空洞卷积后特征图也会发生改变
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
                      weight_std=self.weight_std)
        )

        self.seg_x4 = nn.Sequential(
            ConResAtt(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            ConResAtt(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            ConResAtt(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1)      #
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, 2, kernel_size=1)
        )

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

    def forward(self, x_list):

        x, x_res_d, x_res_h, x_res_w = x_list
        dim = len(x.shape)
        if dim == 4:
            x = x.unsqueeze(0)
            x_res_d = x_res_d.unsqueeze(0)
            x_res_h = x_res_h.unsqueeze(0)
            x_res_w = x_res_w.unsqueeze(0)
        ## encoder   2 是batch_size
        x = self.conv_1_32(x)
        x = self.layer0(x)  # [2,32,128,128,128]
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)  # [2,64,64,64,64]
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)  # [2,128,32,32,32]
        skip3 = x

        x = self.conv_128_256(x)  # [2,256,16,16,16]
        x = self.layer3(x)  # [2,256,16,16,16]
        # 加了一个空洞卷积
        x = self.layer4(x)  # [2,256,16,16,16]
        x = self.fusionConv(x)  # [2,128,16,16,16]

        x_e = x  # 中间结果储存在x_e中，输出去

        ## decoder
        res_x4_d = F.interpolate(x_res_d, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)		# [2,1,32,32,32]
        res_x4_h = F.interpolate(x_res_h, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)		# [2,1,32,32,32]
        res_x4_w = F.interpolate(x_res_w, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)		# [2,1,32,32,32]
        seg_x4 = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)			# [2,128,32,32,32]
        
        # 使用加权层（conv包括卷积、组归一化和RELU激活）来细化Iseg和Iskip的元素求和 Seg_x4就代表F
        seg_x4 = seg_x4 + skip3 #[2,128,32,32,32]
        seg_x4, res_x4_d, res_x4_h, res_x4_w = self.seg_x4([seg_x4, res_x4_d, res_x4_h, res_x4_w])				# [2,64,32,32,32]

        res_x2_d = F.interpolate(res_x4_d, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)			# [2,64,64,64,64]
        res_x2_h = F.interpolate(res_x4_h, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)			# [2,64,64,64,64]
        res_x2_w = F.interpolate(res_x4_w, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)			# [2,64,64,64,64]
        seg_x2 = F.interpolate(seg_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)			# [2,64,64,64,64]
       	
        # 保持通道数相同
        skip3_1 = self.conv_128_64(skip3)
        
        # 多尺度
        # 将skip3上采样一次，然后和skip2一起与segx2相加
        multiscale_x3_1 = F.interpolate(skip3_1, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        
        seg_x2 = seg_x2 + skip2 + multiscale_x3_1		#[2,64,64,64,64]
        seg_x2, res_x2_d, res_x2_h, res_x2_w = self.seg_x2([seg_x2, res_x2_d, res_x2_h, res_x2_w])				# [2,32,64,64,64]

        res_x1_d = F.interpolate(res_x2_d, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)			# [2,32,128,128,128]
        res_x1_h = F.interpolate(res_x2_h, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)			# [2,32,128,128,128]
        res_x1_w = F.interpolate(res_x2_w, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)			# [2,32,128,128,128]
        seg_x1 = F.interpolate(seg_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)			# [2,32,128,128,128]
        
        # 保持通道数相同
        skip3_2 = self.conv_64_32(skip3_1)
        
        skip2_1 = self.conv_64_32(skip2)	# [2,32,64,64,64]
        # 多尺度
        # 将skip3_1再上采样一次，将skip2上采样一次，然后和skip1一起与segx1相加
        multiscale_x3_2 = F.interpolate(skip3_2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)	#[2,32,128,128,128]
        multiscale_x2_1 = F.interpolate(skip2_1, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)	#[2,32,128,128,128]
        
        seg_x1 = seg_x1 + skip1 + multiscale_x3_2 + multiscale_x2_1 #[2,32,128,128,128]
        seg_x1, res_x1_d, res_x1_h, res_x1_w = self.seg_x1([seg_x1, res_x1_d, res_x1_h, res_x1_w])				# [2,32,128,128,128]

        seg = self.seg_cls(seg_x1)  # [2,2,128,128,128]
        res_d = self.res_cls(res_x1_d)  # [2,2,128,128,128]
        res_h = self.res_cls(res_x1_h)  # [2,2,128,128,128]
        res_w = self.res_cls(res_x1_w)  # [2,2,128,128,128]

        resx2_d = self.resx2_cls(res_x2_d)  # [2,2,64,64,64]
        resx2_h = self.resx2_cls(res_x2_h)  # [2,2,64,64,64]
        resx2_w = self.resx2_cls(res_x2_w)  # [2,2,64,64,64]

        resx4_d = self.resx4_cls(res_x4_d)  # [2,2,32,32,32]
        resx4_h = self.resx4_cls(res_x4_h)  # [2,2,32,32,32]
        resx4_w = self.resx4_cls(res_x4_w)  # [2,2,32,32,32]   2 是batch_size

        resx2_d = F.interpolate(resx2_d, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]
        resx2_h = F.interpolate(resx2_h, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]
        resx2_w = F.interpolate(resx2_w, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]

        resx4_d = F.interpolate(resx4_d, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]
        resx4_h = F.interpolate(resx4_h, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]
        resx4_w = F.interpolate(resx4_w, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                                mode='trilinear', align_corners=True)  # [2,2,128,128,128]
        # out_list = [seg.unsqueeze(0), res_d.unsqueeze(0), res_h.unsqueeze(0), res_w.unsqueeze(0), resx2_d.unsqueeze(0),
                    # resx2_h.unsqueeze(0), resx2_w.unsqueeze(0), resx4_d.unsqueeze(0), resx4_h.unsqueeze(0), resx4_w.unsqueeze(0)]

        return x_e, [seg_x1, res_d, res_h, res_w, resx2_d, resx2_h, resx2_w, resx4_d, resx4_h, resx4_w]


