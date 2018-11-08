"""
Dataset preparation of coco

we will using coco to do detections

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


class CoCoDetDataset(data.Dataset):

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

        # transform img and get box label
        return img

    def __len__(self):
        return len(self.img_ids)
