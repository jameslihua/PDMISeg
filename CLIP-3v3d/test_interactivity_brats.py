import argparse
import sys
from PIL import Image
import CLIP.clip as clip
sys.path.append("./CLIP")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from models.CLIP_3v3d_brats import *
from dataset.brats2019.BraTSDataSet import BraTSValDataSet
import os
import os.path as osp
from math import ceil
import nibabel as nib


def get_arguments():
    parser = argparse.ArgumentParser(description="ConResNet for 3D medical image segmentation.")
    parser.add_argument("--data-dir", type=str, default='dataset/brats2019/',
                        help="Path to the directory containing your dataset.")
    parser.add_argument("--data-list", type=str, default='val_list.txt',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default='128,128,128',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes to predict.")
    parser.add_argument("--restore-from", type=str, default='snapshots_brats/Generator40000.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--weight-std", type=bool, default=True,
                        help="whether to use weight standarization in CONV layers.")
    parser.add_argument('--backbone', default='3v3d', help='backbone 3v3d')
    parser.add_argument('--trans_encoding', default='word_embedding',
                        help='the type of encoding: rand_embedding or word_embedding!!!!!!!!!!!!!!!!!!!')
    parser.add_argument('--word_embedding', default='pretrained_weights/txt_encoding_brats.pth',
                        help='The path of word embedding')
    return parser.parse_args()


def pad_image(img, target_size):  # target_size = input_size
    """Pad an image up to the target size.将图像填充到目标尺寸。 target.shape"""
    # bs, channel, depth, heigt, width = img.shape
    # target_size = (d, h, w)
    deps_missing = target_size[0] - img.shape[1]
    rows_missing = target_size[1] - img.shape[2]
    cols_missing = target_size[2] - img.shape[3]
    # 四维填充，左右上下; mode：’constant‘,指的是常量模式
    padded_img = np.pad(img, ((0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


# 滑动预测
def predict_sliding(net, img_list, tile_size, classes):
    image, image_res_d, image_res_h, image_res_w = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)  # 三线性插值上采样
    image_size = image.shape
    #print(image_size)
    overlap = 1 / 3

    strideW = ceil(tile_size[2] * (1 - overlap))
    strideH = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[1] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[2] - tile_size[1]) / strideH) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[3] - tile_size[2]) / strideW) + 1)
    full_probs = torch.zeros((classes, image_size[1], image_size[2], image_size[3]))
    count_predictions = torch.zeros((classes, image_size[1], image_size[2], image_size[3]))

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideH)
                x1 = int(col * strideW)
                d2 = min(d1 + tile_size[0], image_size[1])
                y2 = min(y1 + tile_size[1], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, d1:d2, y1:y2, x1:x2]
                img_res_d = image_res_d[:, d1:d2, y1:y2, x1:x2]
                img_res_h = image_res_h[:, d1:d2, y1:y2, x1:x2]
                img_res_w = image_res_w[:, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_img_res_d = pad_image(img_res_d, tile_size)
                padded_img_res_h = pad_image(img_res_h, tile_size)
                padded_img_res_w = pad_image(img_res_w, tile_size)
                padded_prediction = net([torch.from_numpy(padded_img).cuda(), torch.from_numpy(padded_img_res_d).cuda(),
                                         torch.from_numpy(padded_img_res_h).cuda(),
                                         torch.from_numpy(padded_img_res_w).cuda()])  # [1,2,128,128,128]
                padded_prediction = F.sigmoid(padded_prediction[0])  # [1,2,128,128,128]

                padded_prediction = interp(padded_prediction).cpu().data[0]  # [2,128,128,128]
                prediction = padded_prediction[0:img.shape[1], 0:img.shape[2], 0:img.shape[3], :]  # [2,128,128,128]
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

def id2trainId(label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

def truncate(MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI
    
def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)  # input_size: 110,128,128

    #model = conresnet(input_size, NoBottleneck, [1, 2, 2, 2, 2], num_classes=args.num_classes, weight_std=True)
    model = CLIP_3v3d_brats(img_size=input_size,
                                in_channels=4,
                                out_channels=args.num_classes,
                                backbone=args.backbone,
                                encoding=args.trans_encoding
                                )
    model = nn.DataParallel(model)  # 多GPU训练

    if os.path.exists(args.restore_from):
        model.module.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
        print('loading model successfully')
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))

    model.eval()
    model.cuda()
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    ####
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_model, preprocess = clip.load("./pretrained_weights/ViT-B-32.pt", device=device)
    text_model.load_state_dict(torch.load('./pretrained_weights/text_encoder20000.pth', map_location=torch.device('cpu')), False)
    text_model.eval()
    text_model.cuda()
    
    ORGAN_NAME = ["enhanced tumor", "whole tumor", "tumor core"]  # 分别代表  
    text_ORGAN = torch.cat([clip.tokenize(f'Computed tomography of the {item}') for item in ORGAN_NAME]).to(device)
    standard_text = text_model.encode_text(text_ORGAN) # 3 256 ORGAN_NAME = ["enhanced tumor", "whole tumor", "tumor core"]
    _ = input("Please choose the query organ：enhanced tumor, whole tumor, tumor core\n")
    if _ not in ORGAN_NAME:
        print('input error！')

    input_text = text_model.encode_text(clip.tokenize('Computed tomography of the '+ _).to(device))
    
    '''for item in ORGAN_NAME:
        print(f'Computed tomography of the {item}')     
    print('Computed tomography of the '+ _)'''
    
    simi_to_ET = torch.sqrt(torch.sum((input_text - standard_text[0, :]) ** 2))  # 计算欧氏距离
    simi_to_WT = torch.sqrt(torch.sum((input_text - standard_text[1, :]) ** 2))
    simi_to_TC = torch.sqrt(torch.sum((input_text - standard_text[2, :]) ** 2))
    
    '''print(input_text)
    print(standard_text[2, :])
    
    print(simi_to_ET)
    print(simi_to_WT)
    print(simi_to_TC)'''
    
    if min(simi_to_ET, simi_to_WT, simi_to_TC) == simi_to_ET:
        flag = 'ET'
        print('output organ：' + flag)
    if min(simi_to_ET, simi_to_WT, simi_to_TC) == simi_to_WT:
        flag = 'WT'
        print('output organ：' + flag)
    if min(simi_to_ET, simi_to_WT, simi_to_TC) == simi_to_TC:
        flag = 'TC'
        print('output organ：' + flag)
    ####
    
    ######读取数据
    img_ids = [i_id.strip().split() for i_id in open('dataset/brats2019/val_list.txt')] #测试集列表
    test_img = img_ids[0]   # 随机选取第一个数据，可修改
    filepath = test_img[0] +'/'+ osp.splitext(osp.basename(test_img[0]))[0]
    flair_path = filepath + '_flair.nii.gz'
    t1_path = filepath + '_t1.nii.gz'
    t1ce_path = filepath + '_t1ce.nii.gz'
    t2_path = filepath + '_t2.nii.gz'
    label_path = filepath + '_seg.nii.gz'
    name = osp.splitext(osp.basename(filepath))[0]
    flair_file = osp.join('dataset/brats2019/', flair_path)
    t1_file = osp.join('dataset/brats2019/', t1_path)
    t1ce_file = osp.join('dataset/brats2019/', t1ce_path)
    t2_file = osp.join('dataset/brats2019/', t2_path)
    label_file = osp.join('dataset/brats2019/', label_path)
    
    flairNII = nib.load(flair_file)
    t1NII = nib.load(t1_file)
    t1ceNII = nib.load(t1ce_file)
    t2NII = nib.load(t2_file)
    labelNII = nib.load(label_file)
    
    flair = truncate(flairNII.get_fdata())
    t1 = truncate(t1NII.get_fdata())
    t1ce = truncate(t1ceNII.get_fdata())
    t2 = truncate(t2NII.get_fdata())
    image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
    label = labelNII.get_fdata()
    
    label = id2trainId(label)

    image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
    label = label.transpose((0, 3, 1, 2))     # Depth x H x W
    image = image.astype(np.float32)
    label = label.astype(np.float32)

    size = image.shape[1:]
    affine = labelNII.affine

    # image -> res
    cha, dep, hei, wid = image.shape
    image_copy_d = np.zeros((cha, dep, hei, wid)).astype(np.float32)
    image_copy_h = np.zeros((cha, dep, hei, wid)).astype(np.float32)
    image_copy_w = np.zeros((cha, dep, hei, wid)).astype(np.float32)

    image_copy_d[:, 3:, :, :] = image[:, 0:dep - 3, :, :]
    image_copy_h[:, :, 1:, :] = image[:, :, 0:hei - 1, :]
    image_copy_w[:, :, :, 1:] = image[:, :, :, 0:wid - 1]

    image_res_d = image - image_copy_d
    image_res_h = image - image_copy_h
    image_res_w = image - image_copy_w

    image_res_d[:, 0:3, :, :] = 0
    image_res_h[:, :, 0, :] = 0
    image_res_w[:, :, :, 0] = 0

    image_res_d = np.abs(image_res_d)
    image_res_h = np.abs(image_res_h)
    image_res_w = np.abs(image_res_w)
    ######读取数据结束
    
    #size = size[0]
    #affine = affine[0].astype()
    with torch.no_grad():
        output = predict_sliding(model, [image,image_res_d, image_res_h, image_res_w], input_size, args.num_classes)
    '''
    # np.around(X)，对X四舍五入保留整数部分
    # np.asarray 和 np.array 的功能相同，将输入转为矩阵格式
    '''
    seg_pred_3class = np.asarray(np.around(output), dtype=np.uint8)
    seg_gt = np.asarray(label, dtype=int)
    
    if flag == 'ET':
        seg_pred = seg_pred_3class[:, :, :, 0]
        seg_gt = seg_gt[0, :, :, :]
    if flag == 'WT':
        seg_pred = seg_pred_3class[:, :, :, 1]
        seg_gt = seg_gt[1, :, :, :]
    if flag == 'TC':
        seg_pred = seg_pred_3class[:, :, :, 2]
        seg_gt = seg_gt[2, :, :, :]

    ####   
    dice = dice_score(seg_pred[None, :, :, :], seg_gt[None, :, :, :])
    Jaccard = jaccard(seg_pred[None, :, :, :], seg_gt[None, :, :, :])
    Recall = recall(seg_pred[None, :, :, :], seg_gt[None, :, :, :])

    seg_pred = seg_pred.transpose((1,2,0))
    seg_pred = seg_pred.astype(np.int16)
    seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
    seg_save_p = os.path.join('outputs/%s.nii' % (name))
    nib.save(seg_pred, seg_save_p)
    
    print('The {} of {} as shown'.format(flag, name))
    print('Dice_score = {:.4}'.format(dice))
    print('Jaccard_score = {:.4}'.format(Jaccard))
    print('Recall_score = {:.4}'.format(Recall))


if __name__ == '__main__':
    main()
