"""

Segmentation dataset for pytorch

Cityscapes

this file will load cityscapes segmentation data as dataset
official Cityscapes labels are:


labels = [
#       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


"""
from __future__ import print_function

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import glob
import cv2


num_class = 33
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of three channels in the order of BGR
h, w = 1024, 2048
train_h = int(h / 2)  # 512
train_w = int(w / 2)  # 1024
val_h = h  # 1024
val_w = w  # 2048


class CityScapesSegDataset(Dataset):

    def __init__(self, root_dir, target_size, n_class=num_class, is_test=False, crop=False, flip_rate=0.):

        self.image_files = glob.glob(os.path.join(root_dir, 'leftImg8bit/train/*/*.png'))
        self.label_files = [os.path.join(os.path.dirname(i).replace('leftImg8bit', 'gtFine'),
                                         os.path.basename(i).replace('leftImg8bit.png', 'gtFine_labelIds.png'))
                            for i in self.image_files]

        self.image_files = sorted(self.image_files)
        self.label_files = sorted(self.label_files)
        assert len(self.image_files) == len(self.label_files), 'images and label not equal.'
        self.means = means
        self.n_class = n_class
        self.target_size = target_size

        self.is_test = is_test
        self.flip_rate = flip_rate
        self.crop = crop
        if not self.is_test:
            self.crop = True
            self.flip_rate = 0.5

            if len(target_size) > 1:
                self.new_h = target_size[0]
                self.new_w = target_size[1]
            else:
                self.new_h = target_size
                self.new_w = target_size

        print('Dataset load done. {} images, {} targets.'.format(len(self.image_files), len(self.label_files)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img = scipy.misc.imread(img_name, mode='RGB')

        label_name = self.label_files[idx]
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        # 30% probability to crop image
        if self.crop and random.random() < 0.3:
            h, w, _ = img.shape
            top = random.randint(0, h - self.new_h)
            left = random.randint(0, w - self.new_w)
            img = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]
        else:
            # resize it, opencv resize is opposite, if you want (500, 800), send it (800, 500)
            img = cv2.resize(img, dsize=(self.new_w, self.new_h))
            label = cv2.resize(label, dsize=(self.new_w, self.new_h))

        if random.random() < self.flip_rate:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # process image
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # expand label into [n_classes, h, w]
        target = np.zeros([self.n_class, self.new_h, self.new_w])
        for i in range(self.n_class):
            a = (label == i).astype(np.int8)
            target[i] = a

        # cv2.imshow('1', target[26])
        # cv2.waitKey(0)
        img = np.asarray(img, dtype=np.float)
        target = np.asarray(target, dtype=np.double)

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        target = torch.from_numpy(target.copy()).double()

        print(img.type())
        print(target.type())
        # X: img, Y: target (expand to n_class dim) , l: original label (single channel)
        sample = {'X': img, 'Y': target, 'l': label}
        return sample
