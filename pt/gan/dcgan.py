import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


class Generator(nn.Module):

    def __init__(self, n_z, n_g_filters):
        """
        this is the implementation of generator
        in DCGAN, the input a Z: 100,
        final output would be 3x64x64

        so how to achieve that?
        simply just follow the net flow
        """
        super(Generator, self).__init__()
        self.n_z = n_z
        self.n_g_filters = n_g_filters

        self.main = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_g_filters*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_g_filters*8),
            nn.ReLU(True),
            # now size: ngf*8x4x4 = 512x4x4, should upgrade into 1024x4x4?
            nn.ConvTranspose2d(n_g_filters*8, n_g_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_g_filters*4),
            nn.ReLU(True),
            # now size: 256x8x8
            nn.ConvTranspose2d(n_g_filters*4, n_g_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_g_filters*2),
            nn.ReLU(True),
            # now size: 128x16x16
            nn.ConvTranspose2d(n_g_filters*2, n_g_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_g_filters),
            nn.ReLU(True),
            # now size 64x32x32
            nn.ConvTranspose2d(n_g_filters, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, inp):
        return self.main(inp)


class Discriminator(nn.Module):

    def __init__(self, n_d_filters):
        super(Discriminator, self).__init__()
        self.n_d_filtes = n_d_filters
        self.main = nn.Sequential(
            nn.Conv2d(3, n_d_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # now size: 64x32x32
            nn.Conv2d(n_d_filters, n_d_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_d_filters*2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: 128x16x16
            nn.Conv2d(n_d_filters*2, n_d_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_d_filters*4),
            nn.LeakyReLU(0.2, inplace=True),

            # now size: 256x8x8
            nn.Conv2d(n_d_filters*4, n_d_filters*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_d_filters*8),
            nn.LeakyReLU(0.2, inplace=True),
            # now 512x512x4x4
            nn.Conv2d(n_d_filters*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        return self.main(inp)
