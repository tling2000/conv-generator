import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet,CircuConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger,save_image,plot_fft,get_mean_freq_scale
from dat import get_gaussian_2d,get_tiny_imagenet

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


if __name__ == '__main__':
    seed = 0
    device = 'cuda:1'
    dat_mean = 0
    dat_std = 0.1
    sample_num = 100
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    tag = 'mid_dim'
    save_root = f'/data2/tangling/conv-generator/outs/remark3/0915_1/{tag}'
    
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

    inputs  = get_tiny_imagenet(sample_num).to(device)
    for i in range(10):
        image = inputs[i].mean(0).detach()
        save_image(os.path.join(save_path,f'output{i}'),image,f'input')
        low_freq_scale = plot_fft(os.path.join(save_path,f'f_output{i}'),image,f'f_input')
        logger.info(f'sample_{i},input,low_freq_scale:{low_freq_scale}')

    #sample from diff input
    outputs = conv_net(inputs)
    for i in range(10):
        image = outputs[i].mean(0).detach()
        save_image(os.path.join(save_path,f'output{i}'),image,f'output')
        low_freq_scale = plot_fft(os.path.join(save_path,f'f_output{i}'),image,f'f_output')
        logger.info(f'sample_{i},output,low_freq_scale:{low_freq_scale}')

    in_scale = get_mean_freq_scale(inputs.mean(1))
    out_scale = get_mean_freq_scale(outputs.mean(1))
    logger.info(f'input,low_freq_scale:{in_scale}')
    logger.info(f'output,low_freq_scale:{out_scale}')
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