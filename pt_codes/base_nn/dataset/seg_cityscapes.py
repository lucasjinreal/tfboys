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
import PIL
from PIL import Image
from collections import namedtuple
from torchvision.transforms import ToTensor


Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    'id',  # An integer ID that is associated with this label.
    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!
    'category',  # The name of the category that this label belongs to
    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.
    'hasInstances',  # Whether this label distinguishes between single instances or not
    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not
    'color',  # The color of this label
])

# change 255 to 19,
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 19, 'vehicle', 7, False, True, (0, 0, 142)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------
# Please refer to the main method below for example usages!
# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
id2trainId = {label.id: label.trainId for label in reversed(labels)}


def map_func(x):
    return id2trainId[x]


class CityscapesSegDataset(Dataset):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, num_classes=19, split='train', target_size=(512, 1024), transform=False):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val'], 'only train or val support'
        self._transform = transform
        self.num_classes = num_classes
        self.target_size = target_size

        self.image_files = glob.glob(os.path.join(root, 'leftImg8bit/{}/*/*.png'.format(self.split)))
        self.image_files = sorted(self.image_files)
        self.label_files = [os.path.join(os.path.dirname(i).replace('leftImg8bit', 'gtFine'),
                                         os.path.basename(i).replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))
                            for i in self.image_files]

        assert len(self.image_files) == len(self.label_files), 'images and label not equal.'
        print('Found all {} images.'.format(len(self.image_files)))

        # apply label pixel-wise map from id to trainId
        # self.vf = np.vectorize(map_func)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        try:
            img_file = self.image_files[index]

            img = cv2.imread(img_file)
            img = cv2.resize(img, self.target_size)
            img = np.array(img, dtype=np.uint8)

            # load label
            lbl_file = self.label_files[index]
            lbl = PIL.Image.open(lbl_file)
            lbl = lbl.resize(self.target_size)
            # lbl is label, should map it into trainId
            # 0~18, and 255 pixel value
            lbl = np.array(lbl, dtype=np.int32)
            # lbl = np.array(self.vf(lbl), dtype=np.uint8)

            if self._transform:
                return self.transform(img, lbl)
            else:
                return img, lbl
        except Exception as e:
            print('Reading item corrupt: {}'.format(e))
            print('Corrupt file path: ', img_file)
            img_file = self.image_files[index]

            img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.uint8)

            # load label
            lbl_file = self.label_files[index]
            lbl = PIL.Image.open(lbl_file)
            lbl = np.array(lbl, dtype=np.int32)
            # lbl = np.array(self.vf(lbl), dtype=np.uint8)

            if self._transform:
                return self.transform(img, lbl)
            else:
                return img, lbl

    def preprocess(self, img):
        """
        img reading from PIL.Image
        :param img:
        :return:
        """
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        return img

    def transform(self, img, lbl):
        # print('img: {}, lbl: {}'.format(img.shape, lbl.shape))
        # resize img and lbl to target_size
        img = cv2.resize(img, self.target_size)
        lbl = cv2.resize(lbl, self.target_size)

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl





class CityscapesSegDatasetTrainID(Dataset):

    def __init__(self, root, phase='train', num_classes=3, target_size=(512, 1024), subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.phase = phase
        self.target_size = target_size
        self.num_classes = num_classes
        self.images_root += subset
        self.labels_root += subset

        self.EXTENSIONS = ['.jpg', '.png']
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.filtered_classes_map = {
            # road
            0: 1,
            # person
            11: 2,
            # rider
            12: 3,
            # car
            13: 4,
            # truck
            14: 5
            # others, all -1
        }

        print(self.images_root)
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in
                          fn if self.is_image(f)]
        self.filenames.sort()

        # [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        # self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if self.is_label(f)]
        self.filenamesGt.sort()

        self.vfunc = np.vectorize(self.map_func)

    def map_func(self, x):
        if x in self.filtered_classes_map.keys():
            x = self.filtered_classes_map[x]
        else:
            x = 0
        return x

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(self.image_path_city(self.images_root, filename), 'rb') as f:
            image = self.load_image(f).convert('RGB')
            image = np.array(image)
        with open(self.image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = self.load_image(f).convert('P')
            lbl = np.array(label)

        img = cv2.resize(image, self.target_size)
        lbl = cv2.resize(lbl, self.target_size)

        lbl = np.array(self.vfunc(lbl), dtype=np.int32)
        # some ignore, so as white edge
        lbl[lbl == 255] = -1
        # print('999 max: {}, min: {}'.format(np.max(lbl), np.min(lbl)))

        # cv2.imshow('lbl', label)
        # cv2.waitKey(0)

        if self.phase == 'train':
            img, label = self.transform(img, lbl)
        return img, label

    def __len__(self):
        return len(self.filenames)

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    @staticmethod
    def load_image(file):
        return Image.open(file)

    def is_image(self, filename):
        return any(filename.endswith(ext) for ext in self.EXTENSIONS)

    @staticmethod
    def is_label(filename):
        return filename.endswith("_labelTrainIds.png")
        # return filename.endswith("_labelTrainIds.png")

    @staticmethod
    def image_path_city(root, name):
        return os.path.join(root, f'{name}')

