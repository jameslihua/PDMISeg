import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Dice损失   https://blog.csdn.net/weixin_38410551/article/details/105227216

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth  # 平滑变量，防止分母为0
        self.p = p  # 平方值
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        # 分子
        num = torch.sum(torch.mul(predict, target), dim=1)
        # 分母
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        dice_score = 2*num / den
        loss_avg = 1 - dice_score.mean()

        return loss_avg

class DiceLossADNI(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLossADNI, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index  # 需要忽略的类索引 https://blog.csdn.net/weixin_42287851/article/details/99419883

    def forward(self, predict, target):
        # predict和target两者形状相同，否则报错
        assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (predict.shape, target.shape)
        # 对BinaryDiceLoss进行实例化
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.sigmoid(predict)

        for i in range(target.shape[1]):  # label的shape[1]是h:240
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])  # [:,i]取所有维度的第i个元素
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/(target.shape[1]-1 if self.ignore_index!=None else target.shape[1])

# BCE : Binary Cross Entropy
# 添加二分类交叉熵损失函数。
# 在数据较为平衡的情况下有改善作用，但是在数据极度不均衡的情况下，交叉熵损失会在几个训练之后远小于Dice 损失，效果会损失。

class BCELossADNI(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELossADNI, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            # torch.clamp:将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
            output = torch.clamp(output, min=1e-7, max=1-1e-7)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()

# 边界损失
# 迫使模型注意物体表面边界的像素。ConResNet中的边界损失相对于原论文，将损失合并到模型中，与上下文残差学习一起相互促进。
class BCELossBoud(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        bs, category, depth, width, heigt = target.shape  # batch_size, channel, depth, width, height
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:,i]
            targ_i = target[:,i]
            tt = np.log(depth * width * heigt / (target[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        return total_loss