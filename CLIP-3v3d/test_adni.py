import argparse
import sys
from PIL import Image
sys.path.append("./CLIP")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from models.CLIP_3v3d_adni import *
from dataset.ADNI.ADNI1DataSet import ADNI1DataSet, ADNI1ValDataSet
import os
import os.path as osp
from math import ceil
import nibabel as nib


def get_arguments():
    parser = argparse.ArgumentParser(description="ConResNet for 3D medical image segmentation.")
    parser.add_argument("--data-dir", type=str, default='dataset/ADNI/',
                        help="Path to the directory containing your dataset.")
    parser.add_argument("--data-list", type=str, default='val_list.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default='128,128,128',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (L_HP, R_HP).")
    parser.add_argument("--restore-from", type=str, default='snapshots_adni/Generator40000.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--weight-std", type=bool, default=True,
                        help="whether to use weight standarization in CONV layers.")
    parser.add_argument('--backbone', default='3v3d', help='backbone 3v3d')
    parser.add_argument('--trans_encoding', default='word_embedding',
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='pretrained_weights/txt_encoding_adni.pth',
                        help='The path of word embedding')
    return parser.parse_args()


def pad_image(img, target_size):  # target_size = input_size
    """Pad an image up to the target size.将图像填充到目标尺寸。 target.shape"""
    # bs, channel, depth, heigt, width = img.shape
    # target_size = (d, h, w)
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    # 四维填充，左右上下; mode：’constant‘,指的是常量模式
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


# 滑动预测
def predict_sliding(net, img_list, tile_size, classes):
    image, image_res_d, image_res_h, image_res_w = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)  # 三线性插值上采样
    image_size = image.shape
    overlap = 1 / 3

    strideW = ceil(tile_size[2] * (1 - overlap))
    strideH = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideH) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]))

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideH)
                x1 = int(col * strideW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res_d = image_res_d[:, :, d1:d2, y1:y2, x1:x2]
                img_res_h = image_res_h[:, :, d1:d2, y1:y2, x1:x2]
                img_res_w = image_res_w[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_img_res_d = pad_image(img_res_d, tile_size)
                padded_img_res_h = pad_image(img_res_h, tile_size)
                padded_img_res_w = pad_image(img_res_w, tile_size)
                padded_prediction = net([torch.from_numpy(padded_img).cuda(), torch.from_numpy(padded_img_res_d).cuda(),
                                         torch.from_numpy(padded_img_res_h).cuda(),
                                         torch.from_numpy(padded_img_res_w).cuda()])  # [1,2,128,128,128]
                padded_prediction = F.sigmoid(padded_prediction[0])  # [1,2,128,128,128]

                padded_prediction = interp(padded_prediction).cpu().data[0]  # [2,128,128,128]
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], 0:img.shape[4], :]  # [2,128,128,128]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    full_probs = full_probs.numpy().transpose(1, 2, 3, 0)
    return full_probs


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1)

    dice = 2 * num / den

    return dice.mean()


def jaccard(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"

    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1)

    jaccard = num / (den - num)

    return jaccard.mean()


def recall(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"

    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    label_sum = np.sum(target, axis=1)

    recall = num / label_sum

    return recall.mean()


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    d, h, w = map(int, args.input_size.split(','))

    input_size = (d, h, w)  # input_size: 110,128,128
    #model = conresnet(input_size, NoBottleneck, [1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=True)
    model = Universal_model(img_size=input_size,
                                in_channels=1,
                                out_channels=args.num_classes,
                                backbone=args.backbone,
                                encoding=args.trans_encoding
                                )
    model = nn.DataParallel(model)  # 多GPU训练

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        model.module.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))

    model.eval()
    model.cuda()
    testloader = data.DataLoader(
        ADNI1ValDataSet(args.data_dir, args.data_list),
        batch_size=1, shuffle=False, pin_memory=True)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    dice_L_HP = 0
    dice_R_HP = 0
    Jaccard_L_HP = 0
    Jaccard_R_HP = 0
    Recall_L_HP = 0
    Recall_R_HP = 0

    for index, batch in enumerate(testloader):
        image, image_res_d, image_res_h, image_res_w, label, size, name, affine = batch
        size = size[0].numpy()
        affine = affine[0].numpy()
        with torch.no_grad():
            output = predict_sliding(model,
                                     [image.numpy(), image_res_d.numpy(), image_res_h.numpy(), image_res_w.numpy()],
                                     input_size, args.num_classes)
        '''
        # np.around(X)，对X四舍五入保留整数部分
        # np.asarray 和 np.array 的功能相同，将输入转为矩阵格式
        '''
        seg_pred_class = np.asarray(np.around(output), dtype=np.uint8)

        seg_pred_L_HP = seg_pred_class[:, :, :, 0]
        seg_pred_R_HP = seg_pred_class[:, :, :, 1]
        
       

        seg_pred = np.zeros_like(seg_pred_L_HP)
        seg_pred = np.where(seg_pred_L_HP == 1, 21, seg_pred)
        seg_pred = np.where(seg_pred_R_HP == 1, 20, seg_pred)

        seg_gt = np.asarray(label[0].numpy(), dtype=int)

        seg_gt_L_HP = seg_gt[0, :, :, :]
        seg_gt_R_HP = seg_gt[1, :, :, :]

        dice_L_HP_i = dice_score(seg_pred_L_HP[None, :, :, :], seg_gt_L_HP[None, :, :, :])
        dice_R_HP_i = dice_score(seg_pred_R_HP[None, :, :, :], seg_gt_R_HP[None, :, :, :])
        Jaccard_L_HP_i = jaccard(seg_pred_L_HP[None, :, :, :], seg_gt_L_HP[None, :, :, :])
        Jaccard_R_HP_i = jaccard(seg_pred_R_HP[None, :, :, :], seg_gt_R_HP[None, :, :, :])
        Recall_L_HP_i = recall(seg_pred_L_HP[None, :, :, :], seg_gt_L_HP[None, :, :, :])
        Recall_R_HP_i = recall(seg_pred_R_HP[None, :, :, :], seg_gt_R_HP[None, :, :, :])

        print(
            'Processing {}: Dice_L_HP = {:.4}, Dice_R_HP = {:.4}, Jaccard_L_HP = {:.4}, Jaccard_R_HP = {:.4}, Recall_L_HP = {:.4}, Recall_R_HP = {:.4}'.format(
                name, dice_L_HP_i, dice_R_HP_i, Jaccard_L_HP_i, Jaccard_R_HP_i, Recall_L_HP_i, Recall_R_HP_i))

        dice_L_HP += dice_L_HP_i
        dice_R_HP += dice_R_HP_i
        Jaccard_L_HP += Jaccard_L_HP_i
        Jaccard_R_HP += Jaccard_R_HP_i
        Recall_L_HP += Recall_L_HP_i
        Recall_R_HP += Recall_R_HP_i

        seg_pred = seg_pred.transpose((1, 2, 0))

        seg_pred = seg_pred.astype(np.int16)

        seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
        seg_save_p = os.path.join('outputs/%s.nii' % (name[0]))
        nib.save(seg_pred, seg_save_p)

    dice_L_HP_avg = dice_L_HP / (index + 1)
    dice_R_HP_avg = dice_R_HP / (index + 1)
    Jaccard_L_HP_avg = Jaccard_L_HP / (index + 1)
    Jaccard_R_HP_avg = Jaccard_R_HP / (index + 1)
    Recall_L_HP_avg = Recall_L_HP / (index + 1)
    Recall_R_HP_avg = Recall_R_HP / (index + 1)

    print('Average score: Dice_L_HP = {:.4}, Dice_R_HP = {:.4}'.format(dice_L_HP_avg, dice_R_HP_avg))
    print('Average score: Jaccard_L_HP = {:.4}, Jaccard_R_HP = {:.4}'.format(Jaccard_L_HP_avg, Jaccard_R_HP_avg))
    print('Average score: Recall_L_HP = {:.4}, Recall_R_HP = {:.4}'.format(Recall_L_HP_avg, Recall_R_HP_avg))


if __name__ == '__main__':
    main()
