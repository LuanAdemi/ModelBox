import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = conv_block(7, 32)
        self.conv2 = conv_block(32, 64, pool=False)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv3 = conv_block(64, 128, pool=False)
        self.conv4 = conv_block(128, 256, pool=False)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.conv5 = conv_block(256, 128, pool=False)
        self.conv6 = conv_block(128, 64, pool=False)
        self.res3 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.conv7 = conv_block(64, 32, pool=False)
        self.conv8 = conv_block(32, 4, pool=False)
        self.res4 = nn.Sequential(conv_block(4, 4), conv_block(4, 4))


    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        out = self.conv5(out)
        out = self.conv6(out)
        out = F.max_pool2d(self.res3(out) + out, (2,2))

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.res4(out) + out

        return torch.softmax(out.view(4,16),dim=1).view(4,4,4)
