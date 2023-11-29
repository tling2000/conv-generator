from base64 import decode, encode
from turtle import forward
import torch.nn.functional as F
from torchvision.models import resnet18,vgg16,vgg16_bn

import torch
import torch.nn as nn
import numpy as np

from config import IN_CHANNELS,MID_CHANNELS,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM
from coef import kernel_fft

class SingleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pad : int,
        pad_mode: str,
        with_bias: bool,
        with_relu: bool,
        ) -> None:
        super(SingleConv,self).__init__()
        self.pad = pad
        self.pad_mode = pad_mode
        self.with_relu = with_relu
        
        if pad_mode == 'circular_one_side':
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,bias=with_bias)
        else:
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=pad,bias=with_bias,padding_mode=pad_mode)
        if with_relu:
            self.relu = nn.ReLU()

    def forward(self,x):
        if self.pad_mode == 'circular_one_side':
            x = self.conv(F.pad(x,(0,self.pad,0,self.pad),mode='circular'))
        else:
            x = self.conv(x)
        if not self.with_relu:
            return x
        x = self.relu(x)
        return x

class ToyConvAE(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        mid_channels: int,
        conv_num: int,
        ) -> None:
        super(ToyConvAE,self).__init__()

        pad_mode = 'circular_one_side'
        pad = kernel_size - 1
        with_bias = False
        with_relu = True

        if conv_num == 1:
            self.main = nn.Sequential(
            SingleConv(in_channels,in_channels,kernel_size,pad,pad_mode,with_bias,False),
            )
        else:
            self.main = nn.Sequential(
                SingleConv(in_channels,mid_channels,kernel_size,pad,pad_mode,with_bias,with_relu),
                *[SingleConv(mid_channels,mid_channels,kernel_size,pad,pad_mode,with_bias,with_relu) for i in range(conv_num-2)],
                SingleConv(mid_channels,in_channels,kernel_size,pad,pad_mode,with_bias,False),
            )

    def forward(self,x):
        x = self.main(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        mid_channels: int,
        conv_num: int,
        pad : int,
        pad_mode: str = 'circular_one_side',
        with_bias: bool = False,
        with_relu: bool = False,
        ) -> None:
        super(ConvNet,self).__init__()
        assert conv_num >= 0,'the block num must larger than 0'
        assert pad_mode in ['zeros', 'reflect', 'replicate','circular','circular_one_side'],'undefined padding mod'

        self.with_bias = with_bias
        self.conv_num = conv_num

        if conv_num == 1:
            self.main = nn.Sequential(
            SingleConv(in_channels,in_channels,kernel_size,pad,pad_mode,with_bias,False),
            )
        else:
            self.main = nn.Sequential(
                SingleConv(in_channels,mid_channels,kernel_size,pad,pad_mode,with_bias,with_relu),
                *[SingleConv(mid_channels,mid_channels,kernel_size,pad,pad_mode,with_bias,with_relu) for i in range(conv_num-2)],
                SingleConv(mid_channels,in_channels,kernel_size,pad,pad_mode,with_bias,False),
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


class ToyAE(nn.Module):
    def __init__(
        self,
        kernel_size : int,
        in_channels: int,
        mid_channels: int,
        with_upsample: bool,
        with_bias: bool,
        ) -> None:
        super(ToyAE,self).__init__()

        assert kernel_size % 2 == 1
        if with_upsample:
            stride = 2
        else:
            stride = 1
        padding = kernel_size // 2
        out_padding = 2 * padding + stride - kernel_size
        padding_mode = 'zeros'

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode),
            nn.ReLU(),
            nn.Conv2d(mid_channels,mid_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode),
            nn.ReLU(),
            nn.Conv2d(mid_channels,mid_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(mid_channels,mid_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode,output_padding=out_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels,mid_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode,output_padding=out_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels,in_channels,kernel_size,stride,padding,bias=with_bias,padding_mode=padding_mode,output_padding=out_padding),
        )

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class ConvAE(nn.Module):
    def __init__(
        self,
        encode_net: str,
        pretrained: bool,
        image_shape: list,
        ) -> None:
        super(ConvAE,self).__init__()
        assert encode_net in ['vgg16','vgg16_bn','resnet18']
        H,W = image_shape
        assert H in [32,64,224],'only few shape is supported'

        if encode_net == 'vgg16':
            self.encode = vgg16(pretrained=pretrained)
        elif encode_net == 'vgg16_bn':
            self.encode = vgg16_bn(pretrained=pretrained)
        elif encode_net == 'resnet18':
            self.encode = resnet18(pretrained=pretrained)
        else:
            raise RuntimeError()

        mid_dim = 1024
        temp_dim = mid_dim
        modules = []
        modules.append(nn.ConvTranspose2d(1000, mid_dim, 4, 1, 0, bias=True)) #(mid_dim,4,4)
        modules.append(nn.BatchNorm2d(mid_dim))
        modules.append(nn.ReLU())
        
        if H == 32:
            for i in range(3):
                modules.append(nn.ConvTranspose2d(temp_dim, temp_dim // 2, 4, 2, 1, bias=True)) #(mid_dim,4,4)
                modules.append(nn.BatchNorm2d(temp_dim // 2))
                modules.append(nn.ReLU())
                temp_dim = temp_dim // 2 
            modules.append(nn.Conv2d(temp_dim, 3, 3, 1, 1, bias=True))

        elif H == 64:
            for i in range(4):
                modules.append(nn.ConvTranspose2d(temp_dim, temp_dim // 2, 4, 2, 1, bias=True)) #(mid_dim,4,4)
                modules.append(nn.BatchNorm2d(temp_dim // 2))
                modules.append(nn.ReLU())
                temp_dim = temp_dim // 2 
            modules.append(nn.Conv2d(temp_dim, 3, 3, 1, 1, bias=True))


        elif H == 224:
            modules.append(nn.ConvTranspose2d(temp_dim, temp_dim // 2, 3, 2, 1 , bias=True)) #(mid_dim,4,4)
            modules.append(nn.BatchNorm2d(temp_dim // 2))
            modules.append(nn.ReLU())
            temp_dim = temp_dim // 2 
            for i in range(5):
                modules.append(nn.ConvTranspose2d(temp_dim, temp_dim // 2, 4, 2, 1, bias=True)) #(mid_dim,4,4)
                modules.append(nn.BatchNorm2d(temp_dim // 2))
                modules.append(nn.ReLU())
                temp_dim = temp_dim // 2 
            modules.append(nn.Conv2d(temp_dim, 3, 3, 1, 1, bias=True))


        self.decode = nn.Sequential(*modules)

    def forward(self,x):
        x = self.encode(x)
        x = x.view(-1,1000,1,1)
        x = self.decode(x)
        return x


if __name__ == '__main__':
    input = torch.randn(10,3,64,64)
    net = ConvAE(
        encode_net='vgg16',
        pretrained=True,
        image_shape=(224,224)
    )
    output = net(input)
    print(net)


    # a = torch.randn((3,3,2,4))
    # b = torch.randn((3,3,4,6))
    # c = torch.matmul(a,b)
    # print(c.shape)
