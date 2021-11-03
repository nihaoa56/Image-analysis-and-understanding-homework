# 参考https://github.com/meteorshowers/RCF-pytorch
from skimage.color import gray2rgb
from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
from os import listdir
#图像格式转换
def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im
#图像格式转换
def prepare_image_cv2(im):
    im = cv2.resize(im, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im
class RCFLoader(data.Dataset):
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False,sunFileRoot = 'cvtoOut/'):
        self.root = root
        self.split = split
        self.sunFileRoot = sunFileRoot
        self.transform = transform
        self.bsds_root = join(root, 'HED-BSDS')
        self.sunFileList = []
        if self.split == 'train':#BSDS数据
            self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':#BSDS数据
            self.filelist = join(self.bsds_root, 'test.lst')
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        if split=='use':#这里使用自己的图像数据
            #原本的数据集中存在list文件，自己数据一般没有因此动态生成list
            self.listdir(sunFileRoot,self.sunFileList)
            print(self.sunFileList)

    def listdir(self,path, list_name):  #传入存储的list
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)
    def __len__(self):#pytorch的dataloader来获得自定义数据集长度
        return len(self.sunFileList)
    def __getitem__(self, index):#pytorch的dataloader输入一个小于长度的整数来获得自定义数据集对应元素
        if self.split == "train":#训练时使用的部分，基本未作修改
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = cv2.resize(lb, (256, 256), interpolation=cv2.INTER_LINEAR)
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<64)] = 2
            lb[lb >= 64] = 1
        else:
            #根据pytorch的dataloder提供的序号得到对应的文件名
            img_file = self.sunFileList[index]
        if self.split == "train":#训练时代码，未作修改
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            #根据文件地址得到对应图像
            original_img = np.array(cv2.imread(join(img_file)), dtype=np.float32)
            #图像格式转换
            img = prepare_image_cv2(original_img)
            original_img = original_img.transpose(2, 0, 1)
            return img, original_img, img_file

