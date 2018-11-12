"""
Simple predict code on prediction on single image
loading the pretrained RFBNet Mobile detection model.

I shall upgrade MobileNet to MobileNetV2 version


"""
from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot, COCOroot
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512, \
    COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
import cv2



class RFBNetDetector(object):

    def __init__(self, model_type, model_path, input_size=300):
        self.model_type = model_type
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = 81

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _init_net(self):
        if self.model_type == 'RFB_vgg':
            from models.RFB_Net_vgg import build_net
        elif self.model_type == 'RFB_mobile':
            from models.RFB_Net_mobile import build_net
            self.cfg = COCO_mobile_300
            self.input_size = 300

        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
            self.priors = self.priors.to(self.device)

        self.net = build_net('test', num_classes=self.num_classes)
        self.net.load_state_dict(self.model_path)
        self.net.eval()
        self.net.to(self.device)
        print('Finish loading model.')
        rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[self.model_type == 'RFB_mobile']
        self.transform = BaseTransform(self.net.size, rgb_means, (2, 0, 1))
        self.detector = Detect(self.num_classes, 0, self.cfg)
        print('Detector ready.')

    def predict_from_file(self, img_f):
        img = cv2.imread(img_f, cv2.IMREAD_COLOR)
        x = self.transform(img_f).unsqueeze(0).to(self.device)
        out = self.net(x)
        boxes, scores = self.detector.forward(out, self.priors)
        print('boxes: ', boxes)
        print('scores: ', scores)


if __name__ == '__main__':
    detector = RFBNetDetector(model_type='RFB_mobile', model_path='',)
    detector.predict_from_file('')

