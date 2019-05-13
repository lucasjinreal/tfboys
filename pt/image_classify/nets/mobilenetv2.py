"""
implementation of MobileNetV2

the essential of mobilenet is the DepthWise Conv and Width Multiplier
which all using for reduce the amount of network params.

to calculate output of convolution, do:

out_w = (in_w + 2*pad_w - filter_w)/stride + 1
out_h = (in_h + 2*pad_h - filter_h)/stride + 1

so, 224 -> (3*3 2) -> 111

"""
import torch
from torch import nn
import math


# define some util blocks
def conv_bn(x, output, stride):
    return nn.Sequential(
        nn.Conv2d(x, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(x, output):
    return nn.Sequential(
        nn.Conv2d(x, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):

    def __init__(self, x, output, stride, expand_ratio):
        """
        this is the core of MobileNet, something just like ResNet
        but not very much alike.

        it is called Inverted Residual, the opposite residual operation,
        how does it operation anyway?

        Only when stride == 1 && input == output, using residual connect
        other wise normal convolution

        what does this expand_ratio for? this value is the middle expand ratio when you transfer
        input channel to output channel ( you will get a middle value right? so there it is)

        :param x:
        :param output:
        :param stride:
        :param expand_ratio:
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], 'InsertedResidual stride must be 1 or 2, can not be changed'
        self.user_res_connect = self.stride == 1 and x == output

        # this convolution is the what we called Depth wise separable convolution
        # consist of pw and dw process, which is transfer channel and transfer shape in 2 steps
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(x, x * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(x * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(x * expand_ratio, x * expand_ratio, 3, stride, 1, groups=x*expand_ratio, bias=False),
            nn.BatchNorm2d(x*expand_ratio),
            nn.ReLU6(inplace=True),
            # pw linear
            nn.Conv2d(x*expand_ratio, output, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        if self.user_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    implementation of MobileNetV2
    """

    def __init__(self, num_classes=20, input_size=224, width_mult=1.):
        """
        we just need classes, input_size and width_multiplier here

        the input_size must be dividable by 32, such as 224, 480, 512, 640, 960 etc.
        the width multiplier is width scale which can be 1 or less than 1
        we will judge this value to determine the last channel and input channel
        but why?

        Here is the reason for this:
        You can set input channel to 32, and the output of MobileNetV2 must be 1280
        so, when you multiply that channel, accordingly output should also be multiplied
        :param num_classes:
        :param input_size:
        :param width_mult:
        """
        super(MobileNetV2, self).__init__()

        assert input_size % 32 == 0, 'input_size must be divided by 32, such as 224, 480, 512 etc.'
        input_channel = int(32*width_mult)
        self.last_channel = int(1280*width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]

        # t:  c: channel, n: , s: stride
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c*width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # build last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # this why input must can be divided by 32
        self.features.append(nn.AvgPool2d(int(input_size / 32)))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    n.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()









