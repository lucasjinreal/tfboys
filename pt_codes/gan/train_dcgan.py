"""
this is the file to train on 
DCGAN with a face data
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import os
import numpy as np
import cv2
from alfred.dl.torch.common import device

from dcgan import Generator, Discriminator


data_root = '/media/jintain/wd/permenant/datasets/celeba'
model_save_path = 'log/dcgan_model_d_and_g.pt'
batch_size = 32
image_size = 64
n_latent_vector = 100
n_g_filters = 64
n_d_filters = 64

epochs = 500
lr = 0.0002


def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train():
    os.makedirs('log', exist_ok=True)

    ds = datasets.ImageFolder(root=data_root,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net_g = Generator(n_latent_vector, n_g_filters).to(device)
    net_g.apply(weight_init)

    net_d = Discriminator(n_d_filters).to(device)
    net_d.apply(weight_init)

    if os.path.exists(model_save_path):
        all_state_dict = torch.load(model_save_path)
        net_d.load_state_dict(all_state_dict['d_state_dict'])
        net_g.load_state_dict(all_state_dict['g_state_dict'])
        print('model restored from {}'.format(model_save_path))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, n_latent_vector, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(0.5, 0.999))

    print('start training...')
    
    try:
        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                # update Discrinimator, maximize d loss
                net_d.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=device)
                output = net_d(real_cpu).view(-1)
                err_d_real = criterion(output, label)
                err_d_real.backward()

                d_x = output.mean().item()

                # train with fake batch
                noise = torch.randn(b_size, n_latent_vector, 1, 1, device=device)
                fake = net_g(noise)
                label.fill_(fake_label)
                output = net_d(fake.detach()).view(-1)
                err_d_fake = criterion(output, label)
                err_d_fake.backward()

                d_g_z1 = output.mean().item()
                err_d = err_d_real + err_d_fake
                optimizer_d.step()

                # update Generator
                net_g.zero_grad()
                label.fill_(real_label)
                output = net_d(fake).view(-1)
                err_g = criterion(output, label)
                err_g.backward()
                d_g_z2 = output.mean().item()
                optimizer_d.step()

                if i % 50 == 0:
                    print(f'Epoch: {epoch}, loss_d: {err_d.item()}, loss_g: {err_g.item()}')
        
            if epoch % 2 == 0 and epoch != 0:
                with torch.no_grad():
                    fake = net_g(fixed_noise).detach().cpu().numpy()
                    cv2.imwrite('log/{}_fake.png'.format(epoch), fake)
                    print('record a fake image to local.')
        
    except KeyboardInterrupt:
        print('interrupted, try saving the model')
        all_state_dict = {
            'd_state_dict': net_d.state_dict(),
            'g_state_dict': net_g.state_dict(),
        }
        torch.save(all_state_dict, model_save_path)
        print('model saved...')
        

if __name__ == "__main__":
    train()



