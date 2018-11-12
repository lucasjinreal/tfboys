#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

from nets.seg.fcn8s import FCN8s
from nets.seg.fcn16s import FCN16s
from nets.seg.fcn32s import FCN32s

from segment_trainer import Trainer
from dataset.seg_voc import VOC2012ClassSeg, VOC2011ClassSeg
from torch.utils.data import DataLoader
from alfred.dl.torch.common import device

voc_root = '/media/jintain/sg/permanent/datasets/VOCdevkit'
here = osp.dirname(osp.abspath(__file__))
if not os.path.exists(voc_root):
    print('{} not found.'.format(voc_root))
    exit(0)


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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument(
        '--max-iteration', type=int, default=100000, help='max iteration'
    )
    parser.add_argument(
        '--lr', type=float, default=1.0e-14, help='learning rate',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--pretrained-model',
        default=FCN16s.download(),
        help='pretrained model of FCN16s',
    )
    args = parser.parse_args()
    args.model = 'FCN8s'

    # 1. dataset
    train_loader = DataLoader(VOC2012ClassSeg(root=voc_root, transform=True), batch_size=1, shuffle=True)
    val_loader = DataLoader(VOC2011ClassSeg(root=voc_root, split='seg11valid', transform=True), batch_size=1,
                            shuffle=False)

    # 2. model
    model = FCN8s(n_class=21).to(device)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn16s = FCN16s()
        state_dict = torch.load(args.pretrained_model)
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        model.copy_params_from_fcn16s(fcn16s)

    # 3. optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
