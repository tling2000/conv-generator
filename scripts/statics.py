import torch
import numpy as np
import os
from config import MID_DIM,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM,SAMPLE_NUM,PARAM_MEAN,PARAM_STD
from models import ConvNet
from coefficient import kernel_fft,kernel_mask
from utils import plot_heatmap,set_random,get_delta

if __name__ == '__main__':
    save_path = '/data2/tangling/conv-expression/outs/lab7/statics'
    # seed = 3
    # set_random(seed)

    input = torch.randn((1,MID_DIM,*IMAGE_SHAPE))
    # input = torch.zeros((1,MID_DIM,*IMAGE_SHAPE))
    # input[0,:,0,0] = 1

    out_fft_list = []

    conv_net = ConvNet(
            KERNEL_SIZE,
            MID_DIM,
            CONV_NUM,
            False,
        )
    for i in range(SAMPLE_NUM):

        conv_net.reset_params()
        output = conv_net(input)

        in_image = input.detach() #(1,C,H,W)
        out_image = output.detach() #(1,C,H,W)
        in_fft = torch.fft.fft2(in_image) #(1,C,H,W)
        out_fft = torch.fft.fft2(out_image)
        out_fft_list.append(out_fft)

    out_ffts = torch.concat(out_fft_list)
    out_fft_2s = torch.abs(out_ffts)**2

    #real mean
    mean = torch.mean(out_ffts,dim=0) #(C,H,W)
    mean_2 = torch.mean(out_fft_2s,dim=0) #(C,H,W)

    #cal mean and mean_2
    Ruv = torch.from_numpy(kernel_mask(KERNEL_SIZE,IMAGE_SHAPE,(0,0))) * (IMAGE_SHAPE[0] * IMAGE_SHAPE[1]) #(H,W)
    cal_mean = PARAM_MEAN * Ruv #(H,W)
    cal_mean_2 = torch.abs(PARAM_MEAN * Ruv)**2 + KERNEL_SIZE**2 * PARAM_STD**2 #(H,W)
    #cal mean and mean_2 of T 
    for i in range(1,CONV_NUM):
        cal_mean  = cal_mean * MID_DIM * PARAM_MEAN * Ruv
        cal_mean_2 = cal_mean_2 * MID_DIM * (torch.abs(PARAM_MEAN * Ruv)**2 + KERNEL_SIZE**2 * PARAM_STD**2) + (MID_DIM-1)/MID_DIM * torch.abs(cal_mean)**2
    #cal mixed product of g
    mixed_product= torch.ones(IMAGE_SHAPE,dtype=torch.complex64) #(H,W)
    for i in range(IMAGE_SHAPE[0]):
        for j in range(IMAGE_SHAPE[1]):
            mixed_product[i,j] = (torch.conj(in_fft[0,:,i,j]).reshape(-1,1)@ in_fft[0,:,i,j].reshape(1,-1)).sum()
    mixed_product = torch.real(mixed_product - torch.sum(torch.abs(in_fft)**2,dim=(0,1)))

    #cal mean and mean_2 of g
    cal_mean_2 = cal_mean_2 * torch.sum(torch.abs(in_fft)**2,dim=(0,1)) + torch.abs(cal_mean)**2 * mixed_product #(H,W)
    cal_mean = cal_mean * torch.sum(in_fft,dim=(0,1))  #(H,W)

    print(get_delta(mean,cal_mean))
    print(get_delta(mean_2,cal_mean_2))

    plot_heatmap(save_path,torch.abs(mean[0]),'mean')
    plot_heatmap(save_path,torch.abs(cal_mean),'cal_mean')
    plot_heatmap(save_path,torch.abs(mean_2[0]),'mean_2')
    plot_heatmap(save_path,torch.abs(cal_mean_2),'cal_mean_2')

    # print(norm2[1] - norm2[0])


