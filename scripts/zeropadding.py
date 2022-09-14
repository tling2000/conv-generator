import os,sys
sys.path.append('../src')

import torch
from utils import plot_heatmap

def plot_mean_abs(save_path,mat,name):
    assert len(mat.shape) == 3,''
    mean_abs = torch.mean(torch.abs(mat),dim=(0)).detach()
    mean_abs_norm = mean_abs / mean_abs.max()
    plot_heatmap(save_path,mean_abs_norm,name)

if __name__ == '__main__':
    dat_path = '/data2/tangling/conv-generator/data/17flowers/image.pt'
    save_path = '/data2/tangling/conv-generator/temp'

    mean,std = 1,1
    H,W = 32,32
    H_,W_ = 48,48
    images = torch.load(dat_path)
    input = images[0,0]
    input = torch.randn((1000,H,W)) * std + mean
    zero = torch.zeros((1000,H_,W_))
    zero[:,:H,:W] = input

    f_input = torch.fft.fftshift(torch.fft.fft2(input),dim=(-2,-1))
    f_zero = torch.fft.fftshift(torch.fft.fft2(zero),dim=(-2,-1))

    
    # f_input[0,0] = 0
    # f_zero[0,0] = 0
    plot_heatmap(save_path,torch.abs(f_input.mean(dim=0)),'f_input')
    plot_heatmap(save_path,torch.abs(f_zero.mean(dim=0)),'f_zero')
    plot_heatmap(save_path,input.mean(dim=0),'input',vmax=1.1,vmin=0)
    plot_heatmap(save_path,zero.mean(dim=0),'zero',vmax=1.1,vmin=0)