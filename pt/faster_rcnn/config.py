"""
config.py we configuration
the network base params
"""
import numpy as np


class Config(object):

    def __init__(self):
        # anchor scales and ratios
        self.anchor_scales = [8, 16, 32]
        self.anchor_ratios = [0.5, 1, 2]
        self.feature_stride = [16, ]