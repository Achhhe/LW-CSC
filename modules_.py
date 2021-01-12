import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
from functools import reduce


class CALayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(CALayer,self).__init__()
        # global average pooling:feature(H*W*C)->point(1*1*C)
        self.ave_pool   = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,padding=0,bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//reduction,channel,1,padding=0,bias=False),
            nn.Sigmoid())


    def forward(self,inp):
        atte = self.ave_pool(inp)
        atte = self.conv_atten(atte)
        return inp*atte











