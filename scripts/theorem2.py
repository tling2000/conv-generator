import os,sys
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
    device = 'cuda:0'
    dat_mean = 0
    dat_std = 0.01
    sample_num = 1000
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = '/data2/tangling/conv-generator/outs/theorem2'
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')
    assert CONV_NUM == 1,'must be single layer'

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
        #sample the input
        input = get_gaussian_2d(dat_mean,dat_std,size=(1,IN_CHANNELS,*IMAGE_SHAPE)).to(device) #1*C*H*W
        output = conv_net(input) #1*C*H*W
        #real_fft
        f_input = torch.fft.fft2(input)[0].detach() #C*H*W
        f_output = torch.fft.fft2(output)[0].detach() #C*H*W
        #get T and b
        weight = conv_net.main[0].conv.weight.detach()
        bias = conv_net.main[0].conv.bias.detach()
        f_weight =  kernel_fft(weight,IMAGE_SHAPE,device)
        f_bias = bias * (H*W)

        #cal
        cal_f_output = torch.zeros((IN_CHANNELS,H,W),dtype=torch.complex64,device=device) #C*(H-K+1)*(W-K+1)
        #mat multiply
        for u in range(H):
            for v in range(W):
                cal_f_output[:,u,v] = f_weight[:,:,u,v] @ f_input[:,u,v]
        #add the basic frequency
        cal_f_output[:,0,0] += f_bias 

        #get error
        error = get_error(f_output.cpu(),cal_f_output.cpu())
        error_lis.append(error)
    
    mean_error = np.array(error_lis).mean()
    logger.info(f'mean error:{mean_error}')

