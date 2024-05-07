import os.path as osp
import numpy as np

np.set_printoptions(threshold=np.inf)
import re
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize


class ADNI1DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(128,128,128), scale=True, mirror=True,
                 ignore_label=255):
        self.root = root  # 数据集地址
        self.list_path = list_path  # 数据集列表
        self.crop_d, self.crop_h, self.crop_w = crop_size  # 裁剪尺寸
        self.scale = scale  # 是否需要均匀随机采样
        self.ignore_label = ignore_label  # 要忽视的标签索引 好像是没有用到
        self.is_mirror = mirror  # 是否采用镜像
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]  # 列表 存放每张图像及其标签在数据集中的地址

        # 复制数据，训练时根据max_iter数将列表翻倍
        if not max_iters == None:  # max_iters= args.num_steps * args.batch_size
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / (len(self.img_ids) / 2)))

        self.files = []  # 用来放数据的列表

        for item in self.img_ids:  # 遍历每一张训练样本
            pic = re.search('image', str(item))
            if pic != None:
                filepath = osp.splitext(osp.basename(item[0]))[0][0:6]

                img_path = filepath + '_image.nii'
                label_path = filepath + '_mask.nii'

                img_file = osp.join(root, osp.join('ADNI1-M-TRAIN/', img_path))
                label_file = osp.join(root, osp.join('ADNI1-M-TRAIN/', label_path))

                self.files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": filepath
                })
        print('{} train images are loaded!'.format(len(self.img_ids) / 2))

    def __len__(self):  # 数据集长度
        return len(self.files)

    # 将相关病的索引值归到对应的区域
    def id2trainId(self, label):  # ADNI数据集规定:左海马：21  右海马：20  背景：0
        shape = label.shape  # 读取label形状 ADNI的label_shape是(160，192，192) h，w，d
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))  # 创建一个和标签shape相同，但全0的数组(c,h,w,d)

        L_HP = (label == 21)
        R_HP = (label == 20)

        results_map[0, :, :, :] = np.where(L_HP, 1, 0)  # np.where 三个参数满足条件condition时输出x，不满足时输出y
        results_map[1, :, :, :] = np.where(R_HP, 1, 0)

        return results_map

    # 直方图归一化
    def truncate(self, MRI):  # 归一化为标准正态分布
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))  # 返回一个数组，MRI是输入，bins是定义给定范围内的等宽bin
        indexs = np.argwhere(Hist >= 50)  # 返回非0的数组元组的索引，其中括号内参数是要索引数组的条件。
        index_max = np.float32(indexs[-1, 0])  # 输出这个数组索引的最大值
        MRI[np.where(MRI >= index_max)] = index_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):

        datafiles = self.files[index]

        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])

        image = self.truncate(imageNII.get_fdata())[:, :, :, 0]  # image的shape是160，192，192，1，现只取前三项
        image = np.array([image])  # 将image修改为了[image] image: [1,160,192,192]
        label = labelNII.get_fdata()

        image = image.astype(np.float32)  # c,h,w,d  1,160,192,192
        label = label.astype(np.float32)  # h,w,d

        # 对输入的图片进行采样，scale是均匀随机数，crop_size = input_size
        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)  # 在均匀分布中随机采样，左闭右开
        else:
            scaler = 1

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d, = label.shape  # （160,192,192）

        # random.randint：生成一个指定范围内的整数，这两个值怎么确定的？
        d_off = random.randint(30, img_d - 10 - scale_d)
        h_off = random.randint(10, img_h - 10 - scale_h)
        w_off = random.randint(20, img_w - 25 - scale_w)

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]  # 128,128,110
        label = self.id2trainId(label)

        # transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置
        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W

        # 随机镜像翻转
        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if self.scale:
            image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (2, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True,
                           preserve_range=True)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # image -> res
        image_copy_d = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)  # 1,110,128,128
        image_copy_h = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        image_copy_w = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        
        image_copy_d[:, 3:, :, :] = image[:, 0:self.crop_d - 3, :, :]
        image_copy_h[:, :, 3:, :] = image[:, :, 0:self.crop_h - 3, :]
        image_copy_w[:, :, :, 1:] = image[:, :, :, 0:self.crop_w - 1]
        
        image_res_d = image - image_copy_d
        image_res_h = image - image_copy_h
        image_res_w = image - image_copy_w
        
        image_res_d[:, 0:3, :, :] = 0
        image_res_h[:, :, 0:3, :] = 0
        image_res_w[:, :, :, 0] = 0
        
        image_res_d = np.abs(image_res_d)
        image_res_h = np.abs(image_res_h)
        image_res_w = np.abs(image_res_w)

        # label -> res
        label_copy_d = np.zeros((2, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy_h = np.zeros((2, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy_w = np.zeros((2, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        
        label_copy_d[:, 3:, :, :] = label[:, 0:self.crop_d - 3, :, :]
        label_copy_h[:, :, 3:, :] = label[:, :, 0:self.crop_h - 3, :]
        label_copy_w[:, :, :, 1:] = label[:, :, :, 0:self.crop_w - 1]
        
        label_res_d = label - label_copy_d
        label_res_h = label - label_copy_h
        label_res_w = label - label_copy_w
        
        label_res_d = np.abs(label_res_d)
        label_res_h = np.abs(label_res_h)
        label_res_w = np.abs(label_res_w)
        
        label_res_d[np.where(label_res_d == 0)] = 0
        label_res_h[np.where(label_res_h == 0)] = 0
        label_res_w[np.where(label_res_w == 0)] = 0
        
        label_res_d[np.where(label_res_d != 0)] = 1
        label_res_h[np.where(label_res_h != 0)] = 1
        label_res_w[np.where(label_res_w != 0)] = 1
        
        return image.copy(), image_res_d.copy(), image_res_h.copy(), image_res_w.copy(), label.copy(), label_res_d.copy(), label_res_h.copy(), label_res_w.copy()

      


class ADNI1ValDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:  # 遍历每一张训练样本
            pic = re.search('image', str(item))
            if pic != None:
                filepath = osp.splitext(osp.basename(item[0]))[0][0:6]

                img_path = filepath + '_image.nii'
                label_path = filepath + '_mask.nii'

                img_file = osp.join(root, osp.join('ADNI1-M-VAL/', img_path))
                label_file = osp.join(root, osp.join('ADNI1-M-VAL/', label_path))

                self.files.append({
                    "image": img_file,
                    "label": label_file,
                    "name": filepath
                })
        print('{} val images are loaded!'.format(len(self.img_ids)/2))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):  # ADNI数据集规定:左海马：21  右海马：20  背景：0
        shape = label.shape  # 读取label形状 ADNI的label_shape是(160, 192, 192)
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))  # 用0填充形状
        L_HP = (label == 21)
        R_HP = (label == 20)

        results_map[0, :, :, :] = np.where(L_HP, 1, 0)  # np.where 三个参数满足条件condition时输出x，不满足时输出y
        results_map[1, :, :, :] = np.where(R_HP, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)  ### 这个为什么是50
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])

        image = self.truncate(imageNII.get_fdata())[:, :, :, 0]
        image = np.array([image])
        label = labelNII.get_fdata()

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        name = datafiles["name"]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        size = image.shape[1:]
        affine = labelNII.affine

        # image -> res
        cha, dep, hei, wei = image.shape
        image_copy_d = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy_h = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy_w = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy_d[:, 3:, :, :] = image[:, 0:dep - 3, :, :]
        image_copy_h[:, :, 3:, :] = image[:, :, 0:hei - 3, :]
        image_copy_w[:, :, :, 1:] = image[:, :, :, 0:wei - 1]
        image_res_d = image - image_copy_d
        image_res_h = image - image_copy_h
        image_res_w = image - image_copy_w

        image_res_d[:, 0:3, :, :] = 0
        image_res_h[:, :, 0:3, :] = 0
        image_res_w[:, :, :, 0:] = 0
        image_res_d = np.abs(image_res_d)
        image_res_h = np.abs(image_res_h)
        image_res_w = np.abs(image_res_w)

        return image.copy(), image_res_d.copy(), image_res_h.copy(), image_res_w.copy(), label.copy(), np.array(size), name, affine
