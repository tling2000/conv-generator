import os,sys
import torch
import numpy as np
sys.path.append('../src')

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT,SRC_PATH
from models import CircuConvNet
from coef import kernel_fft
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

    save_root = '/data2/tangling/conv-generator/outs/theorem1'
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,None)
    set_random(seed)
    save_current_src(save_path,SRC_PATH)
    assert CONV_NUM == 1,'must be single layer'

    #int the dataset
    inputs = get_gaussian_2d(0,0.01,size=(sample_num,IN_CHANNELS,*IMAGE_SHAPE))

    # init the conv net
    conv_net = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad_mode='none',
        with_bias=WITH_BIAS,
    )

    outputs = conv_net(inputs)
    f_inputs = torch.fft.fft2(inputs)
    f_outputs = torch.fft.fft2(outputs)

    weight = conv_net.main[0].conv.weight.detach()
    bias = conv_net.main[0].conv.bias.detach()

    f_weight =  kernel_fft(weight)
    f_bias = bias * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]

    print(f_weight.shape)
    print(f_inputs.shape)
    print(f_outputs.shape)
    # f_forward = 

    
    # cal_out_ffts =

    # error = get_error(forward_fft.numpy(),out_fft.numpy())
    # error_lis.append(error)
        
    # total = np.array(error_lis).mean()
    # print(total)

