"""
simple classifier for both train and predict
on single image

the classify image data is using flowers
"""
from classifier_trainer import Trainer
from nets.mobilenet_v2 import MobileNetV2
import sys
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch


target_size = 224
num_classes = 5


def test_data():
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )
    # write some data loader
    dataset = ImageFolder('../../tf_codes/data/flower_photos/', transform=transform,)
    print(dataset.class_to_idx)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1)
    for i, batch_data in enumerate(dataloader):
        if i == 0:
            img, target = batch_data
            img = img.numpy()
            target = target.numpy()

            print(img)
            print(target)
            print(img.shape)
            print(target.shape)
            cv2.imshow('rr', np.transpose(img[0], [1, 2, 0]))
            # cv2.imshow('rrppp', np.transpose(target[0], [1, 2, 0]))
            cv2.waitKey(0)


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    # write some data loader
    dataset = ImageFolder('../../tf_codes/data/flower_photos/', transform=transform)
    print(dataset.class_to_idx)
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=0)

    model = MobileNetV2(num_classes=num_classes, input_size=target_size)
    trainer = Trainer(model=model, train_loader=dataloader, val_loader=None, save_epochs=50,
                      checkpoint_dir='./checkpoints', resume_from='checkpoint.pth.tar', 
                      num_epochs=100)
    trainer.train()


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

