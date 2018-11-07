import os

import cv2

voc_root = '/media/jintian/netac/permanent/voc/VOCdevkit/VOC2012'

def read_images(root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClassAug', i+'.png') for i in images]

    img = cv2.imread(data[0], cv2.COLOR_BGR2RGB)
    seg = cv2.imread(label[0], cv2.COLOR_BGR2RGB)
    return img, seg


if __name__ == '__main__':
    x, y = read_images(voc_root)
    print(x, y)

    cv2.imshow('img', x)
    cv2.imshow('img2', y)
    cv2.waitKey(0)