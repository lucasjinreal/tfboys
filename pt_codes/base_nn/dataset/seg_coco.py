"""
Read original image and segmentation
from folder to do segmentation task

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


class CoCoSegDataset(data.Dataset):

    def __init__(self, coco_root, train=True, transforms=None):
        self.coco_root = coco_root
        if not os.path.exists(self.coco_root):
            print('{} not found.'.format(self.coco_root))
            exit(1)
        # using train2014 by default
        # all coco images are jpg format
        self.image_lists = glob.glob(os.path.join(self.coco_root, 'train2014/*.jpg'))
        self.ann_file = os.path.join(self.coco_root, 'annotations', 'instances_train2014.json')

        self.coco = COCO(self.ann_file)
        self.label_idx_map = dict()

        # a list indicates label and it's id
        self.label_idx_list = self.coco.loadCats(self.coco.getCatIds())
        self.img_ids = self.coco.getImgIds()
        print('Find all {} images from coco train2014.'.format(len(self.img_ids)))
        print('Just using coco tools load annotations, but you can also load from your label or folder, simple.')

        if transforms:
            self.transforms = transforms

    def __getitem__(self, img_id):
        current_id = self.img_ids[img_id]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=current_id))

        img_f = os.path.join(self.coco_root, 'train2014', self.coco.loadImgs(current_id)[0]['file_name'])
        img = Image.open(img_f).convert('RGB')

        # convert annotations to a mask
        seg_mask = np.zeros([img.size[0], img.size[1]], dtype=np.int32)
        for seg in annotations:
            # one instance seg
            seg_points = seg['segmentation']
            color = gen_unique_color(seg['category_id'])
            seg_points = np.asarray(seg_points[0], dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(seg_mask, [seg_points], int(seg['category_id']))

        # convert mask to Image object which using in pytorch
        seg_mask = Image.fromarray(seg_mask)

        if self.transforms:
            img = self.transforms(img)
            seg_mask = self.transforms(seg_mask)
        return img, seg_mask

    def __len__(self):
        return len(self.img_ids)
