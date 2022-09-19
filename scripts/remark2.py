import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet,CircuConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger,plot_fft,save_image
from dat import get_tiny_imagenet

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
    seed = 1
    device = 'cuda:1'
    sample_num = 10
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/remark2'
    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    dat = get_tiny_imagenet(sample_num).to(device)

    conv_net = ConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        with_bias=WITH_BIAS,
        with_relu=True
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    inputs = dat.detach()
    outputs = inputs.clone().detach()
    for j in range(10):
        image = inputs[j].mean(0).detach()
        save_image(os.path.join(save_path,f'output{j}'),image,'input')
        plot_fft(os.path.join(save_path,f'f_output{j}'),image,'input')
    for i in range(CONV_NUM):
        outputs = conv_net.main[i](outputs)
        if (i+1) % 5 == 0:
            for j in range(10):
                image = outputs[j].mean(0).detach()
                save_image(os.path.join(save_path,f'output{j}'),image,f'layer{i+1}')
                plot_fft(os.path.join(save_path,f'f_output{j}'),image,f'layer{i+1}')
