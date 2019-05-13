"""
simple classifier for both train and predict
on single image

the classify image data is using flowers


At least 40 epochs you can see converge evidence
Epoch: 43  iter: 100, loss: 0.37397363781929016
Epoch: 43  iter: 200, loss: 0.7303532958030701
Epoch: 43  iter: 300, loss: 0.8449480533599854
Epoch: 44  iter: 100, loss: 0.5675420165061951
Epoch: 44  iter: 200, loss: 0.8853188157081604
Epoch: 44  iter: 300, loss: 0.9220790863037109
Checking prediction ouput...
output vs target:
[0 2 1 0 4 2 3 1 4 3]
[0 2 0 4 4 4 4 1 4 3]
accuracy: 60.0%


"""
from classifier_trainer import Trainer
from nets.mobilenetv2 import MobileNetV2
from nets.mobilenetv3 import MobileNetV3Small

import sys
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import torch
from PIL import Image
import glob
from alfred.dl.torch.common import device


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
    dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, num_workers=1)
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
    dataset = ImageFolder('../../data/flower_photos/', transform=transform)
    print(dataset.class_to_idx)
    num_classes = len(dataset.classes)
    dataloader = DataLoader(dataset=dataset, batch_size=12, shuffle=True, num_workers=0)

    # model = MobileNetV2(num_classes=num_classes, input_size=target_size)
    model = MobileNetV3Small(num_classes=num_classes)
    trainer = Trainer(model=model, train_loader=dataloader, val_loader=None, save_epochs=50,
                      checkpoint_dir='./checkpoints', resume_from='checkpoint.pth.tar', 
                      num_epochs=100)
    trainer.train()


def predict(img_f):
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ]
    )
    dataset = ImageFolder('../../data/flower_photos/', transform=transform)

    # model = MobileNetV2(num_classes=num_classes, input_size=target_size).to(device)
    model = MobileNetV3Small(num_classes=num_classes).to(device)
    model.eval()
    model.load_state_dict(torch.load('./weights/mb3_flowers.pth.tar')['state_dict'])
    print('model loaded.')
    
    if os.path.isdir(img_f):
        all_imgs_f = glob.glob(os.path.join(img_f, '*.jpg'))
        for img_f in all_imgs_f:
            img = Image.open(img_f)
            img_tensor = transform(img)
            print(img_tensor.size())
            # there is a batch norm issue when inference on single image (batch is 1)
            res = model(img_tensor.unsqueeze(0).to(device))
            res = res.detach().cpu().numpy()[0]
            print(res)
            print('this is: {}'.format(dataset.classes[np.argmax(res)]))

            cv2.imshow('rr', np.array(np.array(img)*255, np.uint8))
            cv2.waitKey(0)
    else:
        img = Image.open(img_f)
        img_tensor = transform(img)
        print(img_tensor.size())
        # there is a batch norm issue when inference on single image (batch is 1)
        res = model(img_tensor.unsqueeze(0).to(device))
        res = res.detach().cpu().numpy()[0]
        print(res)
        print('this is: {}'.format(dataset.classes[np.argmax(res)]))



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'predict':
            img_f = sys.argv[2]
            predict(img_f)
        elif sys.argv[1] == 'preview':
            test_data()
    else:
        print('python3 classifier.py train to train net'
              '\npython3 classifier.py predict img_f/path to predict img.')

