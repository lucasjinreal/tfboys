import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

# custom utilization tool
from util import seg_utils as utils
from alfred.dl.torch.common import device


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, model, optimizer,
                 train_loader, val_loader, max_iter, epochs=1000, out='checkpoints/fcn_seg',
                 size_average=False, interval_validate=None):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.start_epoch = 0

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

        self.load_checkpoint()

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = self.val_loader.dataset.num_classes

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):

            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train(self):
        self.model.train()
        n_class = self.train_loader.dataset.num_classes

        try:
            for e in range(self.start_epoch, self.epochs):
                for i, (data, target) in enumerate(self.train_loader):
                    if data is not None and target is not None:

                        data, target = data.to(device), target.to(device)
                        print('data: {}, target: {}'.format(data.size(), target.size()))
                        data, target = Variable(data), Variable(target)
                        self.optimizer.zero_grad()
                        score = self.model(data)

                        loss = cross_entropy2d(score, target, size_average=self.size_average)
                        loss /= len(data)
                        loss_data = loss.data.item()
                        if np.isnan(loss_data):
                            raise ValueError('loss is nan while training')
                        loss.backward()
                        self.optimizer.step()

                        metrics = []
                        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                        lbl_true = target.data.cpu().numpy()
                        acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(
                            lbl_true, lbl_pred, n_class=n_class)
                        metrics.append((acc, acc_cls, mean_iu, fwavacc))
                        metrics = np.mean(metrics, axis=0)

                        if i % 10 == 0:
                            print('Epoch: {}, iter: {}, acc: {}, acc_cls: {}, mean_iou: {}'.format(
                                e, i, acc, acc_cls, mean_iu
                            ))
                        if e % 50 == 0 and e != 0:
                            self.validate()
                    else:
                        print('passing one invalid training sample.')
                        continue
                if e % 10 == 0:
                    self.save_checkpoint(epoch=e, iter=i)
        except KeyboardInterrupt:
            print('Try saving model, pls hold...')
            self.save_checkpoint(epoch=e, iter=i)
            print('Model has been saved into: {}'.format(os.path.join(self.out, 'checkpoint.pth.tar')))

    def save_checkpoint(self, epoch, iter, is_best=True):
        torch.save({
            'epoch': epoch,
            'iteration': iter,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))

    def load_checkpoint(self):
        if not self.out:
            os.makedirs(self.out)
        else:
            filename = os.path.join(self.out, 'checkpoint.pth.tar')
            if os.path.exists(filename) and os.path.isfile(filename):
                print('Loading checkpoint {}'.format(filename))
                checkpoint = torch.load(filename)
                self.start_epoch = checkpoint['epoch']
                # self.best_top1 = checkpoint['best_top1']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
                print('checkpoint loaded successful from {} at epoch {}'.format(
                    filename, self.start_epoch
                ))
            else:
                print('No checkpoint exists from {}, skip load checkpoint...'.format(filename))
