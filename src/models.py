import os
import random
from turtle import forward
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import numpy as np

from config import MID_DIM,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM,PARAM_MEAN,PARAM_STD

class CircuConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        with_bias,) -> None:
        super(CircuConv,self).__init__()
        self.pad = kernel_size-1
        if with_bias:
            self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,bias=False)


    def forward(self,x):
        x = F.pad(x,(0,self.pad,0,self.pad),mode='circular')
        # x = F.pad(x,(1,1,1,1),mode='circular')
        x = self.conv(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        mid_dims: int,
        conv_num: int,
        with_bias: bool
        ) -> None:
        super(ConvNet,self).__init__()
        assert conv_num >= 1,'the block num must larger than 1'

        self.main = nn.Sequential(
            *[CircuConv(mid_dims,mid_dims,kernel_size,with_bias) for i in range(conv_num)],
        )

    def kaiming_uniform_(self,tensor,fan_in,a=0):
        bound = math.sqrt(6/(1+a**2)/fan_in)
        with torch.no_grad():
            return tensor.uniform_(-bound,bound)

    def kaiming_normal_(self,tensor,fan_in,a=0):
        std = math.sqrt(2/(1+a**2)/fan_in)
        with torch.no_grad():
            return tensor.normal_(mean=0,std=std)

    def fixed_normal_(self,tensor,mean,std):
        with torch.no_grad():
            return tensor.normal_(mean=mean,std=std)

    def reset_params(self,):
        for layer_name,layer in self.named_parameters():
            if layer_name.endswith('weight'):
                self.fixed_normal_(layer,PARAM_MEAN,PARAM_STD)

    def forward(self,x):
        x = self.main(x)
        return x

if __name__ == '__main__':
    model = ConvNet(
        KERNEL_SIZE,
        MID_DIM,
        CONV_NUM,
    )
    print(model)