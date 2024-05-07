import torch
def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"

    # 计算Dice前，要保持shape相同
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)  # 因为view()要求Tensor是连续的，因此需要contiguous。

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()
  
def jaccard(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"

    # 计算Dice前，要保持shape相同
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)  # 因为view()要求Tensor是连续的，因此需要contiguous。

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    jaccard = (num) / (den - num)

    return jaccard.mean()
  
def jaccard(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"

    # 计算Dice前，要保持shape相同
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)  # 因为view()要求Tensor是连续的，因此需要contiguous。

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    jaccard = (num) / (den - num)

    return jaccard.mean()
