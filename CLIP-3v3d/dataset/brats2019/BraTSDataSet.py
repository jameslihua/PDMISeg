import os.path as osp
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize

#定义一个可以实例化的类，继承Dataset类
class BraTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(80, 160, 160), scale=True, mirror=True, ignore_label=255):
        self.root = root #图片路径：dataset/MICCAI_BraTS_2018_Data_Training/
        self.list_path = list_path #图片列表 
        self.crop_d, self.crop_h, self.crop_w = crop_size #要裁剪的尺寸
        self.scale = scale  
        self.ignore_label = ignore_label
        self.is_mirror = mirror #是否镜像化
        '''
        str.strip():移除字符串头尾指定的字符（默认为空格或换行符）,该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        str2 = "   Runoob    ";
        print str2.strip();  
        -->Runoob      
        str.spilt():指定分隔符对字符串进行切片
        '''
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)] #获取图像id

        # 复制数据
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids))) #np.ceil()向上取整

        self.files = [] #创建数组，存储数据
        #这个循环获取了Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_flair.nii.gz
        for item in self.img_ids:   
            '''
          osp.basename()返回目标路径中去掉目录的文件名
          osp.basename('D:\\A\\B\\C\\new.txt')
          --> 'new.txt'

          join(path1,path2)，将path1和path2各部分组合成一个路径
          osp.join('D:\\A\\B\\C\\','.\\new.txt')
          -->'D:\\A\\B\\C\\.\\new.txt' 

          splitext(path)，将path的文件名和扩展名分离，返回(f_name, f_extension)元组
          osp.splitext('D:\\A\\B\\C\\.\\new.txt')
          -->('D:\\A\\B\\C\\.\\new', '.txt')
            '''
            filepath = item[0] +'/'+ osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz' #Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_flair.nii.gz
            t1_path = filepath + '_t1.nii.gz'    #Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1_t1.nii.gz
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'

            name = osp.splitext(osp.basename(filepath))[0] # 返回  Brats18_CBICA_AAB_1
            flair_file = osp.join(self.root, flair_path) # 返回  dataset/MICCAI_BraTS_2018_Data_Training/ Brats18_CBICA_AAB_1/ Brats18_CBICA_AAB_1_flair.nii.gz
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)

            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids))) #返回加载数据的数量

    def __len__(self):
        return len(self.files) #__len__()的返回值就是最终Dataset加载出来的数据个数，
    
    # 将相关病的索引值归到对应的区域
    def id2trainId(self, label):
        shape = label.shape #获得标签shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2])) #创建一个和标签shape相同，但全0的数组(c,h,w,d)
        
        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)  
        results_map[1, :, :, :] = np.where(WT, 1, 0)  # WT = ED + ET + NET
        results_map[2, :, :, :] = np.where(TC, 1, 0)  # TC = ET+NET
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))  
        # np.histogram(a,bins)是一个生成直方图的函数，a是待统计数据的数组；bins指定统计的区间个数；,返回两个值，第一个是数组
        
        idexs = np.argwhere(Hist >= 50) 
        '''
        np.argwhere(a),返回数组中非0数组元素的索引，a是索引数组的条件
        x = ([0,1,2],
            [3,4,5])
        np.argwhere(x>1)
        -->([0,2],#元素2的索引
          [1,0],#元素3的索引
          [1,1],#元素4的索引
          [1,2])
        '''
        idex_max = np.float32(idexs[-1, 0]) #取数组index最后一行的第0个元素，转为float,取得这个数组的最大行
        MRI[np.where( MRI>=idex_max )] = idex_max #将这个数组的最大值的索引传给MRI，并输出数组MRI中大于索引值的数组元素。
        #下边这个是啥玩意？
        sig = MRI[0, 0, 0] #将MRI数组中的首个元素标记为sig
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        '''
        np.where(condition,x,y)
        满足条件(condition)，输出x，不满足输出y。只有条件(condition)，没有x和y,输出满足条件的数据的索引
        a = np.array([2,4,6,8,10])
        np.where(a>5) 输出数组a中大于5的数的索引
        -->([2,3,4])
        a[np.where(a>5)] 相当于输出a[a>5]
        -->([6,8,10])
        '''
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index] #获取数据文件
        
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])

        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_fdata())
        t1 = self.truncate(t1NII.get_fdata())
        t1ce = self.truncate(t1ceNII.get_fdata())
        t2 = self.truncate(t2NII.get_fdata())

        image = np.array([flair, t1, t1ce, t2]) #np.array()的作用就是按照一定要求将object转换为数组？  # 4x240x240x150
        label = labelNII.get_fdata()
        
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        # 随机采样
        if self.scale:
            scaler = np.random.uniform(0.9, 1.1) # numpy.random.uniform(low,high,size) 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开
        else:
            scaler = 1

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape 
        d_off = random.randint(0, img_d - scale_d) # random.randint（下限，上限）：在[下限，上限)区间内，随机选一个值
        h_off = random.randint(15, img_h-15 - scale_h)
        w_off = random.randint(10, img_w-10 - scale_w)

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]  # 4维
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]   # 3维

        label = self.id2trainId(label)
        
        #transpose对矩阵做转置 Channel x H x W x Depth --> Channel x Depth x H x W
        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W

        # 随机镜像操作
        if self.is_mirror:
            randi = np.random.rand(1) #返回一个0~1均匀分布的随机样本值
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]  # [::-1]表示逆序
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
        '''
            线性插值
            image:输入图像
            outshape:
            order:插值的顺序
            mode：{'constant', 'edge', 'symmetric', 'reflect', 'wrap'},根据给定模式填充输入边界之外的可选点
            cval: 图像边界
            clip: 布尔值，是否将输出裁剪到输入图像的值范围内
            preserve_range：布尔值，是否保留原始值范围。
        '''
        if self.scale:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # image -> res 残差模块
        image_copy_d = np.zeros((4, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        image_copy_h = np.zeros((4, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        image_copy_w = np.zeros((4, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32) 
        
        image_copy_d[:, 3:, :, :] = image[:, 0:self.crop_d - 3, :, :]
        image_copy_h[:, :, 1:, :] = image[:, :, 0:self.crop_h - 1, :]
        image_copy_w[:, :, :, 1:] = image[:, :, :, 0:self.crop_w - 1]
        
        image_res_d = image - image_copy_d
        image_res_h = image - image_copy_h
        image_res_w = image - image_copy_w
        
        image_res_d[:, 0:3, :, :] = 0
        image_res_h[:, :, 0, :] = 0
        image_res_w[:, :, :, 0] = 0
        
        image_res_d = np.abs(image_res_d)
        image_res_h = np.abs(image_res_h)
        image_res_w = np.abs(image_res_w)

        # label -> res
        label_copy_d = np.zeros((3, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy_h = np.zeros((3, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy_w = np.zeros((3, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        
        label_copy_d[:, 3:, :, :] = label[:, 0:self.crop_d - 3, :, :]
        label_copy_h[:, :, 1:, :] = label[:, :, 0:self.crop_h - 1, :]
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
#加载验证数据集
class BraTSValDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0] +'/'+ osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
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


    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_fdata())
        t1 = self.truncate(t1NII.get_fdata())
        t1ce = self.truncate(t1ceNII.get_fdata())
        t2 = self.truncate(t2NII.get_fdata())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_fdata()
        name = datafiles["name"]

        label = self.id2trainId(label)

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

        return image.copy(), image_res_d.copy(), image_res_h.copy(), image_res_w.copy(), label.copy(), np.array(size), name, affine
