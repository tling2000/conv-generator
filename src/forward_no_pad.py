import torch
import numpy as np
import os
from config import MID_DIM,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM
from models import ConvNet
from kernel import kernel_fft
from utils import plot_heatmap,set_random

if __name__ == '__main__':
    save_path = '/data2/tangling/conv-expression/outs/lab7/forward'
    seed = 0
    set_random(seed)
    conv_net = ConvNet(
        KERNEL_SIZE,
        MID_DIM,
        CONV_NUM,
    )

    lis = []
    for sample in range(1):
        input = torch.randn((1,MID_DIM,*IMAGE_SHAPE))
        output = conv_net(input)

        in_image = input[0].detach() #C*H*W
        out_image = output[0].detach() #C*H*W
        in_fft = torch.fft.fft2(in_image)
        out_fft = torch.fft.fft2(out_image)
        forward_fft = in_fft.clone().detach() #C*H*W

        for i in range(CONV_NUM):

            weight = conv_net.main[i].conv.weight.detach()
            bias = conv_net.main[i].conv.bias.detach()
            bias_fft = bias * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
            weight_fft = kernel_fft(weight,IMAGE_SHAPE)

            for j in range(IMAGE_SHAPE[0]):
                for k in range(IMAGE_SHAPE[1]):
                    if j == 0 and k == 0:
                        forward_fft[:,j,k] = weight_fft[:,:,j,k] @ forward_fft[:,j,k] + bias_fft
                    else:
                        forward_fft[:,j,k] = weight_fft[:,:,j,k] @ forward_fft[:,j,k]

        delta_fft = (forward_fft - out_fft)
        delta_abs_fft = (torch.abs(forward_fft) - torch.abs(out_fft))

        tag = 'cir'
        plot_heatmap(save_path,torch.abs(out_fft[0]),f'out_fft_{tag}')
        plot_heatmap(save_path,torch.abs(delta_fft[0]),f'delta_fft_{tag}',vmin=0,vmax=torch.abs(out_fft).max())
        plot_heatmap(save_path,torch.abs(delta_abs_fft[0]),f'delta_abs_fft_{tag}',vmin=0,vmax=torch.abs(out_fft).max())
        delta_norm = np.linalg.norm(delta_fft.numpy().reshape(-1),ord=2)
        fft_norm = np.linalg.norm(out_fft.numpy().reshape(-1),ord=2)
        lis.append(delta_norm/fft_norm)

    total = np.array(lis).mean()
    print(total)

