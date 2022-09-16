import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet,CircuConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
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

def plot_mean_abs(save_path,mat,name):
    assert len(mat.shape) == 4,''
    mean_abs = torch.mean(torch.abs(mat),dim=(0,1)).detach()
    mean_abs_norm = mean_abs / mean_abs.max()
    plot_heatmap(save_path,mean_abs_norm,name,cbar=False)

if __name__ == '__main__':
    seed = 2
    device = 'cuda:1'
    sample_num = 100
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/remark1/som'
    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    dat = get_tiny_imagenet(sample_num).to(device)

    conv_net = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad_mode='same',
        with_bias=WITH_BIAS,
        with_relu=False
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    #plot1
    som_lis = []
    for i in range(0,50):
        T,_ = conv_net.get_freq_trans(IMAGE_SHAPE,[0,i+1],'cpu')
        som = torch.real(T * torch.conj(T))[0]
        som_lis.append(som)
    som_array = torch.concat(som_lis)

    log_som_array = torch.log10(som_array)
    x = range(1,CONV_NUM+1)
    fig,ax = plt.subplots()
    # ax.set_yscale('log')
    ax.grid(True,which="both", linestyle='--')
    ax.set_ylabel('SOM')
    ax.set_xlabel('layer')
    for i in range(3):
        for j in range(3):
            ax.plot(x,som_array[:,i,j])
    fig.savefig(os.path.join(save_path,'som.jpg'))
    