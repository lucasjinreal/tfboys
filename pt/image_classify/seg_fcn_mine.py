"""
this actually does not work now

doing semantic segmentation


there is a link talk about loss:
http://sshuair.com/2017/10/21/pytorch-loss/

"""

from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from nets.unet import UNet
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, RandomResizedCrop, Resize
from torch.optim import lr_scheduler
import cv2
import torch.nn.functional as F

from dataset.seg_cityscapes import CityScapesSegDataset
from alfred.dl.torch.common import device
from nets.fcns import VGGNet, FCN8s, FCNs

# h, w
target_size = (512, 1024)
# cityscapes classes num
n_classes = 33
epochs = 500
lr = 1e-4
batch_size = 2
save_per_epoch = 5
save_dir = 'checkpoints/seg'

cityscapes_root = '/media/jintain/sg/permanent/datasets/Cityscapes'
# cityscapes_root = '/media/jintian/netac/permanent/datasets/Cityscapes'


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
    train_data = CityScapesSegDataset(root_dir=cityscapes_root, target_size=target_size, n_class=n_classes)
    data_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=1)
    # test_data(data_loader)

    vgg_model = VGGNet(requires_grad=True, remove_fc=True).to(device)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=0, weight_decay=1e-5)
    # adjust lr every 50 steps
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        scheduler.step()

        epoch_loss = 0
        i = 0
        for i, batch_data in enumerate(data_loader):
            inputs = batch_data['X'].to(device)
            labels = batch_data['Y'].to(device)

            outputs = fcn_model(inputs)
            # print(outputs.cpu().detach().numpy())

            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.data[0]))
        print('Epoch {} finished. Average loss: {}\n'.format(epoch, epoch_loss/i))

        if epoch % save_per_epoch == 0 and epoch != 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                torch.save(fcn_model.state_dict(), os.path.join(save_dir, 'seg_{}_{}.pth'.format(epoch, epoch_loss/i)))
                print('Model has been saved.')


def predict():
    pass


if __name__ == '__main__':
    if len(sys.argv) >= 2:

        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            img_f = sys.argv[2]
            predict()
        elif sys.argv[1] == 'preview':
            test_data()
    else:
        print('python3 classifier.py train to train net'
              '\npython3 classifier.py predict img_f/path to predict img.')