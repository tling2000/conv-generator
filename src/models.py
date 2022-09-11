import os
import random
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import numpy as np

from config import IN_CHANNELS,MID_CHANNELS,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM,PARAM_MEAN,PARAM_STD
from coef import kernel_fft

class CircuConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        pad,
        with_bias,
        ) -> None:
        super(CircuConv,self).__init__()
        self.pad = pad
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,bias=with_bias)

    def forward(self,x):
        x = F.pad(x,(0,self.pad,0,self.pad),mode='circular')
        # x = F.pad(x,(1,1,1,1),mode='circular')
        x = self.conv(x)
        return x

class CircuConvNet(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        mid_channels: int,
        conv_num: int,
        pad_mode : str = 'same',
        with_bias: bool = False,
        ) -> None:
        super(CircuConvNet,self).__init__()
        assert conv_num >= 0,'the block num must larger than 0'
        assert pad_mode in ['same','none'],'undefined padding mod'

        self.with_bias = with_bias
        self.conv_num = conv_num
        if pad_mode == 'same':
            pad = kernel_size-1
        else :
            pad = 0

        if conv_num == 1:
            self.main = nn.Sequential(
            CircuConv(in_channels,in_channels,kernel_size,pad,with_bias),
            )
        else:
            self.main = nn.Sequential(
                CircuConv(in_channels,mid_channels,kernel_size,pad,with_bias),
                *[CircuConv(mid_channels,mid_channels,kernel_size,pad,with_bias) for i in range(conv_num-2)],
                CircuConv(mid_channels,in_channels,kernel_size,pad,with_bias),
            )

    def fixed_normal_(self,tensor,mean,std):
        with torch.no_grad():
            return tensor.normal_(mean=mean,std=std)

    def reset_params(self,mean,std):
        for layer_name,layer in self.named_parameters():
            self.fixed_normal_(layer,mean,std)

    def get_freq_trans(
        self,
        image_shape: list,
        lidx: list,
        device: str
        )->list:

        H,W = image_shape
        # if we only want know single layer
        if type(lidx) == int:
            weight = self.main[lidx].conv.weight.detach()
            f_weight =  kernel_fft(weight,image_shape,device)
            if not self.with_bias:
                return f_weight.detach(),None
            bias = self.main[lidx].conv.bias.detach()
            f_bias = (bias * H * W).to(device)
            return f_weight.detach() ,f_bias.detach()
        
        # else 
        weight = self.main[lidx[0]].conv.weight.detach()
        f_weight = kernel_fft(weight,image_shape,device)
        T =  f_weight

        if self.with_bias:
            bias = self.main[lidx[0]].conv.bias.detach()
            f_bias = (bias * H * W).to(device)
            beta = f_bias
    
        for layer_id in range(lidx[0]+1,lidx[1]):
            weight = self.main[layer_id].conv.weight.detach()
            f_weight = kernel_fft(weight,image_shape,device)
            T = (f_weight.permute((2,3,0,1)) @ T.permute((2,3,0,1))).permute((2,3,0,1))

            if self.with_bias:
                bias = self.main[layer_id].conv.bias.detach()
                f_bias = (bias * H * W).to(device)
                beta = f_bias + torch.real(f_weight[:,:,0,0]) @ beta

        if not self.with_bias:
            return T.detach(),None
        return T.detach(),beta.detach()

    def forward(self,x):
        x = self.main(x)
        return x

if __name__ == '__main__':
    model = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        with_bias=True,
    )
    model.reset_params(0.1,0.3)
    model.get_freq_trans((64,64),(0,3),'cpu')
    # a = torch.randn((3,3,2,4))
    # b = torch.randn((3,3,4,6))
    # c = torch.matmul(a,b)
    # print(c.shape)
