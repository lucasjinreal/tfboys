"""
do the RPN proccess
"""
import numpy as np
from torch import nn




self.rpn_conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

# every anchor will have a score, 2 is yes or not objects
self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
self.rpn_cls_score = nn.Conv2d(in_channels=512, out_channels=self.nc_score_out, kernel_size=1, stride=1, pad=0)
rpn_cls_score = self.rpn_cls_score(rpn_conv1)


# layer for boxes, every box have 4 coordinates
self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
self.rpn_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)
rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv1)

