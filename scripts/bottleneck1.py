import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_UPSAMPLE,WITH_BIAS,DATE,MOMENT
from models import ToyAE
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
from dat import get_gaussian_2d,get_tiny_imagenet
from train import train

def make_dirs(save_root):
    exp_name = "-".join([DATE, 
                        MOMENT,
                        f"conv_num{CONV_NUM}",
                        f"K{KERNEL_SIZE}",
                        f"in_channels{IN_CHANNELS}",
                        f"mid_channels{MID_CHANNELS}",
                        f"bias{WITH_BIAS}"])
    save_path = os.path.join(save_root, exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

if __name__ == '__main__':
    seed = 0
    device = 'cuda:1'
    bs = 100
    lr = 0.0001
    rounds = 50
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/bottleneck1'
    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    conv_ae = ToyAE(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        WITH_UPSAMPLE,
        WITH_BIAS,
    )
    # conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    dat = get_tiny_imagenet(1000)
    Xs = dat.detach()
    train(
        conv_ae,
        Xs,
        save_path,
        rounds,
        batch_size=bs,
        lr=lr,
        device=device,
    )

