import torch
import torch.nn as nn

from config import Config

vgg = models.vgg16()
self.rcnn_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

# header network to get the base feature map
base_feature = self.rcnn_base(im_data)
# rpn network should get rois, rpn loss of classes and boxes
rois, rpn_loss_cls, rpn_loss_bbox = self.rcnn_rpn(base_feature, im_info, gt_boxes, num_boxes)

# ... 

