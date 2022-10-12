import torch
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


def conv_out_size(slen, kernel_size, stride=1, padding=0, dilation=1):
    """
    :param slen: Size length of the image. Should be an int.
    :param kernel_size: Int
    :param stride: Int
    :return: The size length of output after convolution
    This function considers 1-dim case.
    """
    return int((slen + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EntryFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_relu = True):
        super(EntryFlowBlock, self).__init__()

        self.layers = []
        if with_relu:
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
        self.net = nn.Sequential(*self.layers)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 2, 0, 1, 1)
        self.out_net = nn.Sequential(self.conv,
                                     nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.net(x) + self.out_net(x)
        return x


class MiddleFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleFlowBlock, self).__init__()

        self.layers = []
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.net(x) + x
        return x


class ExitFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2):
        super(ExitFlowBlock, self).__init__()

        self.layers = []
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(in_channels, out_channels_1, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels_1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(SeparableConv2d(out_channels_1, out_channels_2, kernel_size=3, padding=1))
        self.layers.append(nn.BatchNorm2d(out_channels_2))
        self.layers.append(nn.MaxPool2d((3, 3), stride=(2, 2), padding=1))
        self.net = nn.Sequential(*self.layers)

        self.conv = nn.Conv2d(in_channels, out_channels_2, 1, 2, 0, 1, 1)
        self.out_net = nn.Sequential(self.conv,
                                     nn.BatchNorm2d(out_channels_2))

    def forward(self, x):
        x = self.net(x) + self.out_net(x)
        return x


class Xception_Network(nn.Module):
    def __init__(self, height, width, num_classes):
        super(Xception_Network, self).__init__()
        self.size = (height, width)

        self.pre_conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.size = (conv_out_size(height, 3, 2, 0), conv_out_size(width, 3, 2, 0))
        self.pre_conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.size = (conv_out_size(self.size[0], 3), conv_out_size(self.size[1], 3))
        self.entry_block1 = EntryFlowBlock(64, 128, with_relu=False)
        self.size = (conv_out_size(self.size[0], 1, 2, 0), conv_out_size(self.size[1], 1, 2, 0))
        self.entry_block2 = EntryFlowBlock(128, 256, with_relu=True)
        self.size = (conv_out_size(self.size[0], 1, 2, 0), conv_out_size(self.size[1], 1, 2, 0))
        self.entry_block3 = EntryFlowBlock(256, 728, with_relu=True)
        self.size = (conv_out_size(self.size[0], 1, 2, 0), conv_out_size(self.size[1], 1, 2, 0))
        self.middle_block1 = MiddleFlowBlock(728, 728)
        self.middle_block2 = MiddleFlowBlock(728, 728)
        self.middle_block3 = MiddleFlowBlock(728, 728)
        self.middle_block4 = MiddleFlowBlock(728, 728)
        self.middle_block5 = MiddleFlowBlock(728, 728)
        self.middle_block6 = MiddleFlowBlock(728, 728)
        self.middle_block7 = MiddleFlowBlock(728, 728)
        self.middle_block8 = MiddleFlowBlock(728, 728)
        self.exit_block = ExitFlowBlock(728, 728, 1024)
        self.size = (conv_out_size(self.size[0], 1, 2, 0), conv_out_size(self.size[1], 1, 2, 0))
        self.post_conv1 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1)
        self.post_relu1 = nn.ReLU()

        self.post_conv2 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1)
        self.post_relu2 = nn.ReLU()

        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.size = (conv_out_size(self.size[0], 3), conv_out_size(self.size[1], 3))
        self.ln = nn.Linear(2048, num_classes)
        # self.ln = nn.Linear(2048 * self.size[0] * self.size[1], num_classes)

        self.net = nn.Sequential(
            self.pre_conv1,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.pre_conv2,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.entry_block1,
            self.entry_block2,
            self.entry_block3,
            self.middle_block1,
            self.middle_block2,
            self.middle_block3,
            self.middle_block4,
            self.middle_block5,
            self.middle_block6,
            self.middle_block7,
            self.middle_block8,
            self.exit_block,
            self.post_conv1,
            nn.BatchNorm2d(1536),
            self.post_relu1,
            self.post_conv2,
            nn.BatchNorm2d(2048),
            self.post_relu2,
            self.average_pool,
            nn.Flatten(),
            self.ln,
        )

    def forward(self, x):
        x = self.net(x)
        return x
