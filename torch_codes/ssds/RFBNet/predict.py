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
from data import BaseTransform, COCO_300, COCO_mobile_300
import cv2
from layers.functions import Detect, PriorBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.nms_wrapper import nms
from utils.timer import Timer
from models.RFB_Net_vgg import build_rfb_vgg_net
from models.RFB_Net_mobile import build_rfb_mobilenet
from PIL import Image, ImageDraw
import time


def test(img_path, model_path='weights/RFB_vgg_COCO_30.3.pth'):
    img_path = img_path
    trained_model = model_path
    if torch.cuda.is_available():
        cuda = True
    if 'mobile' in model_path:
        cfg = COCO_mobile_300
    else:
        cfg = COCO_300

    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        if cuda:
            priors = priors.cuda()
    numclass = 81

    img = cv2.imread(img_path)
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    if 'mobile' in model_path:
        net = build_rfb_mobilenet('test', 300, numclass)  # initialize detector
    else:
        net = build_rfb_vgg_net('test', 300, numclass)  # initialize detector

    transform = BaseTransform(net.size, (123, 117, 104), (2, 0, 1))
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        x = Variable(x)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    state_dict = torch.load(trained_model)['state_dict']
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
    net.load_state_dict(new_state_dict)
    net.eval()
    if cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    print('Finished loading model!')
    # print(net)
    detector = Detect(numclass, 0, cfg)

    tic = time.time()
    out = net(x)  # forward pass

    boxes, scores = detector.forward(out, priors)
    print('Finished in {}'.format(time.time() - tic))
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    # Create figure and axes
    # Display the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # scale each detection back up to the image
    for j in range(1, numclass):
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
            print('{}: {}'.format(j, c_bboxes))
            for box in c_bboxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1, 0)
                cv2.putText(img, '{}'.format(j),  (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)
    cv2.imshow('rr', img)
    cv2.waitKey(0)


test('images/COCO_train2014_000000010495.jpg', model_path='weights/rfb_vgg_300_checkpoint.pth.tar')

