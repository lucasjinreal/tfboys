from __future__ import print_function
import os
import pickle
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import BaseTransform, COCO_300
import cv2
from layers.functions import Detect, PriorBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer
from models.RFB_Net_vgg import build_net
from PIL import Image, ImageDraw
import time

from alfred.vis.image.get_dataset_label_map import coco_label_map


class Detector(object):

    def __init__(self, model_path):
        # self.net_name = net_name
        self.model_path = model_path

        self.num_classes = 81
        self.cuda = torch.cuda.is_available()

        self.label_map_list = list(coco_label_map.values())

        self._init_model()

    def _init_model(self):
        if torch.cuda.is_available():
            cuda = True
        cfg = COCO_300
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward()
            if cuda:
                self.priors = priors.cuda()

        self.net = build_net('test', 300, self.num_classes)  # initialize detector
        state_dict = torch.load(self.model_path)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        if cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        else:
            self.net = self.net.cpu()
        print('Finished loading model!')
        # print(net)
        self.detector = Detect(self.num_classes, 0, cfg)

    def predict_on_img(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        transform = BaseTransform(self.net.size, (123, 117, 104), (2, 0, 1))
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            x = Variable(x)
            if self.cuda:
                x = x.cuda()
                scale = scale.cuda()
        tic = time.time()
        out = self.net(x)  # forward pass
        boxes, scores = self.detector.forward(out, self.priors)
        print('Finished in {}'.format(time.time() - tic))
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        return boxes, scores

    def predict_on_video(self, v_f):
        cap = cv2.VideoCapture(v_f)

        while cap.isOpened():
            ok, frame = cap.read()
            if ok:
                img = frame
                boxes, scores = self.predict_on_img(frame)
                # print(boxes.shape)
                # print(scores.shape)
                # scale each detection back up to the image
                tic = time.time()
                for j in range(1, self.num_classes):
                    # print(max(scores[:, j]))
                    inds = np.where(scores[:, j] > 0.6)[0]
                    # conf > 0.6
                    if inds is None:
                        continue
                    c_bboxes = boxes[inds]
                    c_scores = scores[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
                    keep = nms(c_dets, 0.6)
                    c_dets = c_dets[keep, :]
                    c_bboxes = c_dets[:, :4]

                    # print(c_bboxes.shape)
                    # print(c_bboxes.shape[0])
                    if c_bboxes.shape[0] != 0:
                        # print(c_bboxes.shape)
                        # print('{}: {}'.format(j, c_bboxes))
                        for box in c_bboxes:
                            label = self.label_map_list[j-1]
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1, 0)
                            cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0),
                                        1, cv2.LINE_AA)
                # print('post process time: {}'.format(time.time() - tic))
                cv2.imshow('rr', frame)
                cv2.waitKey(1)
            else:
                print('Done')
                exit(0)


if __name__ == '__main__':
    detector = Detector(model_path='weights/RFB_vgg_COCO_30.3.pth')
    detector.predict_on_video('/media/jintain/sg/permanent/datasets/TestVideos/ETH-Bahnhof.mp4')
