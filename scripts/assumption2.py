import os,sys
from turtle import forward
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import CircuConvNet
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

if __name__ == '__main__':
    seed = 0
    device = 'cpu'
    dat_mean = 0
    dat_std = 0.1
    sample_num = 1
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = '/data2/tangling/conv-generator/outs/corollary1'
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    # init the conv net
    conv_net = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad_mode='same',
        with_bias=WITH_BIAS,
    ).to(device)

    error_lis = []
    for sample_id in tqdm(range(sample_num)):
        conv_net.reset_params(PARAM_MEAN,PARAM_STD)

        