import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet,CircuConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
from dat import get_gaussian_2d

def make_dirs(save_root):
    exp_name = "-".join([DATE, 
                        MOMENT,
                        f"conv_num{CONV_NUM}",
                        f"K{KERNEL_SIZE}",
                        f"in_channels{IN_CHANNELS}",
                        f"mid_channels{MID_CHANNELS}",
                        f"bias{WITH_BIAS}",
                        f"mean{PARAM_MEAN}",
                        f"std{PARAM_STD}",])
    save_path = os.path.join(save_root, exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def plot_mean_abs(save_path,mat,name):
    assert len(mat.shape) == 4,''
    mean_abs = torch.mean(torch.abs(mat),dim=(0,1)).detach()
    mean_abs_norm = mean_abs / mean_abs.max()
    plot_heatmap(save_path,mean_abs_norm,name)

if __name__ == '__main__':
    seed = 2
    device = 'cuda:1'
    dat_mean = 0
    dat_std = 0.1
    sample_num = 10
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    tag = 'mu'
    save_root = f'/data2/tangling/conv-generator/outs/remark2/{tag}'
    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    
    # conv_net = ConvNet(
    #     KERNEL_SIZE,
    #     IN_CHANNELS,
    #     MID_CHANNELS,
    #     CONV_NUM,
    #     with_bias=WITH_BIAS,
    #     with_relu=True
    # ).to(device)
    # conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    conv_net = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad_mode='same',
        with_bias=WITH_BIAS,
        with_relu=True
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    inputs = get_gaussian_2d(dat_mean,dat_std,(sample_num,IN_CHANNELS,H,W)).to(device)

    #sample from diff input
    outputs = conv_net(inputs).cpu()
    f_outputs = torch.fft.fft2(outputs)
    f_outputs[:,:,0,0] = 0

    f_outputs_no_zero = f_outputs[:,:,1:,1:]
    
    f_outputs = torch.fft.fftshift(f_outputs,dim=(-1,-2))
    f_outputs_no_zero = torch.fft.fftshift(f_outputs_no_zero,dim=(-1,-2))

    plot_mean_abs(save_path,f_outputs,'f_outputs')
    plot_mean_abs(save_path,f_outputs_no_zero,'f_output_no_zero')
    plot_mean_abs(save_path,outputs,'outputs')

    #sample from diff net
    # input = get_gaussian_2d(dat_mean,dat_std,(1,IN_CHANNELS,H,W)).to(device)
    # output_lis = []
    # for i in tqdm(range(sample_num)):
    #     conv_net.reset_params(PARAM_MEAN,PARAM_STD)
    #     output_lis.append(conv_net(input))
    # outputs = torch.concat(output_lis)
    # f_outputs = torch.fft.fftshift(torch.fft.fft2(outputs),dim=(-1,-2))

    # mean_f_outputs_abs = torch.mean(torch.abs(f_outputs),dim=(0,1)).cpu().detach()
    # mean_f_outputs_abs_norm = mean_f_outputs_abs / mean_f_outputs_abs.max()
    # plot_heatmap(save_path,mean_f_outputs_abs_norm,'mean_net_abs_norm')
        