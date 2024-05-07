from typing import Sequence, Tuple, Type, Union
import sys
sys.path.append("./models")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from Encoder_decoder_brats import *


class CLIP_3v3d_brats(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone='3v3d', encoding='rand_embedding'):   # out_channels = num_clsses
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == '3v3d':
            self.backbone = conresnet(shape=img_size, block=NoBottleneck, layers=[1, 2, 2, 2, 2], weight_std=True)
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(2, 8),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 128),    # 对中间特征x_e进行组归一化，x_e[batch_size(1), 128, 16, 16, 16]
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=0),        # [batch_size, 256, 8, 8, 8]
                torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding

        self.seg_cls_left = nn.Conv3d(32, 1, kernel_size=1)
        self.seg_cls_right = nn.Conv3d(32, 1, kernel_size=1)
        self.seg_cls_third = nn.Conv3d(32, 1, kernel_size=1)
        self.controller_first = nn.Conv3d(512, 512, kernel_size=1, stride=1,
                                    padding=0)
        self.controller = nn.Conv3d(256 + 256, 256, kernel_size=1, stride=1,
                                    padding=0)
        self.proatt_256 = ProAttention(256)
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
            #self.organ_embedding_res = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels


    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()
    
    def forward(self, x_list):
        x_e, out = self.backbone(x_list)  # 中间x_e ， 输出out
        # print('out:', out.shape)
        # x_e [b,128,16,16,16]  out  1个 b 32 128 128 128 , 9个[ b, 2, 128, 128, 128]
        pred = out[0]              
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        # task_encoding torch.Size([32, 256, 1, 1, 1])

        x_feat = self.GAP(x_e.detach())  # [batch_size, 256, 1, 1, 1]
        b = x_feat.shape[0]
        logits_array = []
        p_array = []
        for i in range(b):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.class_num, 1, 1, 1, 1), task_encoding], 1)   # 2 512 1 1 1
            x_cond = self.controller_first(x_cond)      # 2 512 1 1 1
            
            x_cond = F.relu(x_cond)
            
            p = x_cond
            p_array.append(p.view(self.class_num, 512).unsqueeze(0))
            
            x_cond = self.controller(x_cond)           # 2 256 1 1 1
            # print(x_cond.shape)
            x_cond = self.proatt_256(x_cond)           # 2 32 1 1 1
            #x_cond = 1
            #print(x_cond.view(2, 32))
            
            head_inputs = pred[i, :, :, :, :].unsqueeze(0)
            head_inputs = head_inputs.repeat(self.class_num, 1, 1, 1, 1)  # 2 32 128 128 128
            
            _, _, D, W ,H = head_inputs.shape
            
            out_pred = torch.mul(x_cond, head_inputs)
            out_pred_left = self.seg_cls_left(out_pred[0].unsqueeze(0))
            out_pred_right = self.seg_cls_right(out_pred[1].unsqueeze(0))
            out_pred_third = self.seg_cls_third(out_pred[2].unsqueeze(0))
            out_pred = torch.cat((out_pred_left, out_pred_right, out_pred_third), dim=1)
            logits_array.append(out_pred)

        pred = torch.cat(logits_array, dim=0)
        p_out = torch.cat(p_array, dim=0)
        return pred, out, p_out
    
# 提示注意力
class ProAttention(nn.Module):
    def __init__(self, channel, reduction = 16):
        super().__init__()
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), 32))

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        # print(bahs, chs)
        x = self.channel_excitation(x.view(bahs, chs))
        # print(x.shape)
        chn_avg_out = x.view(bahs, 32, 1, 1, 1)
        chn_se = torch.sigmoid(chn_avg_out)
        #r = 0.2
        #chn_se = r * chn_se - r/2 + 1  #将权重的值域改到0.9-1.1
        #print(chn_se.shape)
        #print(x.shape)
        #chn_se = torch.mul(x, chn_se)
        # print(chn_se.shape)  b 256 16 16 16 
        return chn_se

# ----------------------------------------
#             Discriminator
# ----------------------------------------

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        '''
        Discriminator中包含：
        1、3D卷积，kernel_size为4
        2、IN归一化
        3、leakyRelu
        '''
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 3, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(7, 32, normalization=False),		# label的channel为2，image的channel为1.concat之后为3
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )
        self.final = nn.Conv3d(256, 1, 3, padding=1, bias=False)
	
    # 标签和原图结合
    def forward(self, img_A, img_B):
        #print(img_A.shape)
        #print(img_B.shape)
        img_input = torch.cat((img_A, img_B), 1)  # C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼）	# [2,3,128,128,128]
        intermediate = self.model(img_input)  # 做了4个3D卷积提取特征	# [2,256,8,8,8]
        final = self.final(intermediate)		# [2,1,8,8,8]
        return final 
