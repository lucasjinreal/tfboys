"""
Segmentation dataloader load data directly
from folder which contains images and generated
mask images


"""
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
import torch.utils.data as data
import torch
from os import listdir
from PIL import Image
import glob

from pycocotools.coco import COCO
import cv2
import numpy as np

from vision.vis_kit import gen_unique_color
from PIL import Image


class SegFolderDataset(data.Dataset):

    def __init__(self, data_root, train=True, transforms=None):
        self.data_root = data_root

        self.images_list = sorted(glob.glob(os.path.join(self.data_root, 'images', '*.png')))
        self.seg_labels_list = sorted(glob.glob(os.path.join(self.data_root, 'labels', '*gtFine_labelIds.png')))
        assert len(self.images_list) == len(self.seg_labels_list), 'images must same as label.'
        print('Checking image and label index: \nimage: {} \nlabel: {}'.format(self.images_list[0],
                                                                               self.seg_labels_list[0]))
        if transforms:
            self.transforms = transforms

    def __getitem__(self, idx):

        img = Image.open(self.images_list[idx]).convert('RGB')
        # convert annotations to a mask
        seg_mask = Image.open(self.seg_labels_list[idx])

        if self.transforms:
            img = self.transforms(img)
            seg_mask = self.transforms(seg_mask)
        return img, seg_mask

    def __len__(self):
        return len(self.images_list)
