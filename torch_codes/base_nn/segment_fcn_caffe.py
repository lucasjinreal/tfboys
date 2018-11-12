#!/usr/bin/env python
import os
import os.path as osp
import torch
import sys

from nets.seg.fcn8s import FCN8s
from nets.seg.fcn16s import FCN16s
from nets.seg.fcn32s import FCN32s

from segment_trainer import Trainer
from dataset.seg_voc import VOC2012ClassSeg, VOC2011ClassSeg
from torch.utils.data import DataLoader
from alfred.dl.torch.common import device

from PIL import Image

voc_root = '/media/jintain/sg/permanent/datasets/VOCdevkit'
here = osp.dirname(osp.abspath(__file__))
if not os.path.exists(voc_root):
    print('{} not found.'.format(voc_root))
    exit(0)
pre_train = False


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
    model_type = 'FCN8s'
    train_loader = DataLoader(VOC2012ClassSeg(root=voc_root, transform=True), batch_size=1, shuffle=True)
    val_loader = DataLoader(VOC2011ClassSeg(root=voc_root, split='seg11valid', transform=True), batch_size=1,
                            shuffle=False)

    model = FCN8s(n_class=21).to(device)
    start_epoch = 0
    start_iteration = 0

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
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


def predict(img_f):
    # predict on single image
    dataset = VOC2012ClassSeg(root=voc_root, transform=True)

    model = FCN8s(n_class=21).to(device)
    model.eval()

    image = Image.open(img_f)
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
    out = model(torch.Tensor(img).to(device))
    print(out)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            img_f = sys.argv[2]
            predict(img_f=img_f)
        elif sys.argv[1] == 'preview':
            test_data()
    else:
        print('python3 segment_fc_caffe.py train to train net'
              '\npython3 segment_fc_caffe.py predict img_f/path to predict img.')
