"""
doing semantic segmentation


there is a link talk about loss:
http://sshuair.com/2017/10/21/pytorch-loss/

"""

from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from nets.unet import UNet
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, RandomResizedCrop, Resize
import cv2
import torch.nn.functional as F
from nets.common import device


from dataset.seg_dataset_coco import CoCoSegDataset
from dataset.seg_dataset_folder import SegFolderDataset

target_size = 512
epochs = 500
batch_size = 1
save_per_epoch = 5
save_dir = 'weights/seg'


def test_data(data_loader):
    for i, batch_data in enumerate(data_loader):
        if i == 0:
            img, seg_mask = batch_data
            img = img.numpy()
            seg_mask = seg_mask.numpy()

            print(img[0].shape)
            print(seg_mask[0].shape)
            cv2.imshow('rr', np.transpose(img[0], [1, 2, 0]))
            cv2.imshow('rrppp', np.transpose(seg_mask[0], [1, 2, 0]))
            cv2.waitKey(0)


def train():

    transforms = Compose(
        [RandomResizedCrop(size=target_size), ToTensor()]
    )

    # train_data = CoCoSegDataset(coco_root='/media/jintian/netac/permanent/datasets/coco',
    #                             transforms=transforms)
    train_data = SegFolderDataset(data_root='/media/jintian/netac/permanent/datasets/Cityscapes/tiny_cityscapes',
                                  transforms=transforms)
    data_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=1)
    # test_data(data_loader)

    # Unet input may not be 512, it must be some other input
    # change Unet to FCN with VGG model
    model = UNet(colordim=3).to(device)

    # there are some dimension issue about UNet, fix that later
    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0005, lr=0.0001)
    for epoch in range(epochs):
        epoch_loss = 0

        i = 0
        for i, batch_data in enumerate(data_loader):
            img, seg_mask = batch_data
            seg_mask_predict = model(img)
            seg_mask_probs = F.sigmoid(seg_mask_predict)

            seg_mask_predict_flat = seg_mask_probs.view(-1)
            seg_mask_flat = seg_mask.view(-1)

            loss = nn.BCELoss(seg_mask_predict_flat, seg_mask_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss))
        print('Epoch {} finished. Average loss: {}'.format(epoch, epoch_loss/i))

        if epoch % save_per_epoch == 0 and epoch != 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, 'seg_{}_{}.pth'.format(epoch, epoch_loss/i)))
                print('Model has been saved.')


if __name__ == '__main__':
    train()