"""
Class specific doing training work

classifier or others.

"""
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import shutil
import os
import numpy as np


class Trainer(object):

    def __init__(self, model, train_loader, val_loader, save_epochs, **kwargs):
        self.kwargs = kwargs
        self.save_epochs = save_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        assert isinstance(self.train_loader, DataLoader), 'train_loader must be DataLoader instance.'

        self.num_classes = len(self.train_loader.dataset.classes)
        self.val_loader = val_loader

        self.start_epoch = 0

        self.loss = None
        self.optimizer = None
        self._create_optimization()

        self.load_pretrained_model()
        self.load_checkpoint(self.kwargs['resume_from'])

    def train(self):
        print('Start to train...')
        try:
            for e in range(self.start_epoch, self.kwargs['num_epochs']):
                i = 0
                for data, target in self.train_loader:
                    i += 1
                    data, target = data.to(self.device), target.to(self.device)
                    data_var = Variable(data)

                    output = self.model(data_var)

                    # NOTE: you don't need one hot label, just send it, and not need softmax just using raw output
                    # CrossEntropy will solve all that

                    # should target one-hot??? No!
                    loss = self.loss(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if i % 100 == 0:
                        print('Epoch: {}  iter: {}, loss: {}'.format(e, i, loss))

                if e % self.save_epochs == 0:
                    print('Saving checkpoints at epoch: {}'.format(e))
                    self.save_checkpoint(
                        {
                            'epoch': e + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=False
                    )
                if e % 2 == 0:
                    print('Checking prediction ouput...')
                    print('label vs predict:')
                    a = np.array([np.argmax(i) for i in output.detach().cpu().numpy()])
                    b = target.cpu().numpy()
                    print(b)
                    print(a)
                    c = [i for i in a - b if i == 0]
                    print('accuracy: {}%\n'.format((len(c) / len(a)) * 100))

        except KeyboardInterrupt:
            print('Interrupted, saving checkpoints at epoch: {}'.format(e))
            self.save_checkpoint(
                    {
                        'epoch': e + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False
                )

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(self.kwargs['checkpoint_dir'], filename))
        if is_best:
            shutil.copyfile(os.path.join(self.kwargs['checkpoint_dir'], filename),
                            os.path.join(self.kwargs['checkpoint_dir'], 'model_best.pth.tar'))

    def _create_optimization(self):
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
        )

    def load_pretrained_model(self):
        if 'pretrained_path' in self.kwargs.keys():
            print('Loading pretrained weights...')
            pretrained_dict = torch.load(self.kwargs['pretrained_path'])
            self.model.load_state_dict(pretrained_dict)
            print('Pretrained model load successful.')
        else:
            print('No pretrained path provide, skip this step.')

    def load_checkpoint(self, filename):
        if not os.path.exists(self.kwargs['checkpoint_dir']):
            os.makedirs(self.kwargs['checkpoint_dir'])
        else:
            filename = os.path.join(self.kwargs['checkpoint_dir'], filename)
            if os.path.exists(filename) and os.path.isfile(filename):
                print('Loading checkpoint {}'.format(filename))
                checkpoint = torch.load(filename)
                self.start_epoch = checkpoint['epoch']
                # self.best_top1 = checkpoint['best_top1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print('checkpoint loaded successful from {} at epoch {}'.format(
                    filename, self.start_epoch
                ))
            else:
                print('No checkpoint exists from {}, skip load checkpoint...'.format(filename))


