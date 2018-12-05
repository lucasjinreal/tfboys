#!/usr/bin/env python
"""
Train segmentation using FCNs
on cityscapes, we need:

Person
Car
Road
etc.


Cityscapes still not work yet.


"""
import os
import os.path as osp
import torch
import sys

from nets.seg.fcn8s import FCN8s
from nets.seg.fcn16s import FCN16s
from nets.seg.fcn32s import FCN32s

from seg_trainer import Trainer
from dataset.seg_voc import VOC2012ClassSeg, VOC2011ClassSeg

# change dataloader to alfred wrapper
from torch.utils.data import DataLoader
from alfred.dl.torch.common import device

from PIL import Image
import numpy as np
import cv2
from util.seg_utils import draw_seg,draw_seg_by_dataset
from util.get_dataset_colormap import label_to_color_image

import time
import matplotlib.pyplot as plt
from dataset.seg_cityscapes import CityscapesSegDataset, CityscapesSegDatasetTrainID


cityscapes_root = '/media/jintain/sg/permanent/datasets/Cityscapes'
num_classes = 19 + 1
batch_size = 3
pre_train = False
# hw
target_size = (512, 1024)


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s,
        FCN16s,
        FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


def train():
    here = osp.dirname(osp.abspath(__file__))
    if not os.path.exists(cityscapes_root):
        print('{} not found.'.format(cityscapes_root))
        exit(0)

    model_type = 'FCN8s'

    # train_dataset = CityscapesSegDataset(root=cityscapes_root, num_classes=num_classes, split='train', transform=True)
    num_classes = 6
    train_dataset = CityscapesSegDatasetTrainID(root=cityscapes_root, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = CityscapesSegDataset(root=cityscapes_root, num_classes=num_classes, split='val', transform=True)
    val_dataset = CityscapesSegDatasetTrainID(root=cityscapes_root, num_classes=num_classes)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FCN8s(n_class=num_classes).to(device)
    model.train()

    if pre_train:
        fcn16s = FCN16s()
        state_dict = torch.load(fcn16s.download())
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        model.copy_params_from_fcn16s(fcn16s)

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': 10e-4, 'weight_decay': 0},
        ],
        lr=10e-5, momentum=0, weight_decay=0)

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        max_iter=150000,
        interval_validate=4000,
        out='checkpoints/fcn_seg/cityscapes'
    )
    trainer.train()


def predict(img_f):
    # predict on single image, just for preprocess image only
    dataset = CityscapesSegDataset(root=cityscapes_root, split='train', target_size=target_size, transform=True)

    model = FCN8s(n_class=num_classes).to(device)
    model.eval()

    image = Image.open(img_f)
    img_original = np.asarray(image)
    img = dataset.preprocess(image)

    filename = os.path.join('checkpoints/fcn_seg', 'checkpoint.pth.tar')
    if os.path.exists(filename) and os.path.isfile(filename):
        print('Loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('checkpoint loaded successful from {}.'.format(filename))
    else:
        print('No checkpoint exists from {}, skip load checkpoint...'.format(filename))

    # now do predict
    inp = torch.Tensor(img).unsqueeze(dim=0).to(device)
    print('input: ', inp.size())
    out = model(inp)
    out = out.detach().cpu().numpy()[0]
    print(out.shape)

    # convert [n_classes, w, h] to [w, h] mask
    res = np.asarray(np.argmax(out, axis=0), dtype=np.int8)
    print(res.shape)
    print(res)
    # color idx to mask
    masked, mask_color = draw_seg_by_dataset(img_original, res, 'pascal', alpha=0.7)
    cv2.imshow('masked', masked)
    cv2.imshow('masked_color', mask_color)
    cv2.imshow('res', res)

    cv2.imwrite('results/{}_masked.png'.format(time.time()), masked)
    cv2.imwrite('results/{}_mask_color.png'.format(time.time()), mask_color)
    cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            img_f = sys.argv[2]
            predict(img_f=img_f)
        elif sys.argv[1] == 'preview':
            pass
            # test_data()
    else:
        print('python3 segment_fc_caffe.py train to train net'
              '\npython3 segment_fc_caffe.py predict img_f/path to predict img.')
