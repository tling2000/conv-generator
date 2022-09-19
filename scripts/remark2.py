import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger,plot_fft,save_image
from dat import get_data

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
    seed = 2
    device = 'cpu'
    sample_num = None
    with_relu = True

    pad = KERNEL_SIZE  // 2
    pad_mode = 'zeros'

    # pad = KERNEL_SIZE - 1
    # pad_mode = 'circular_one_side'

    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/remark2/'
    data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'

    trace_id = [391,1328,1438,2393,2914,3035,4497,5600]
    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    dat = get_data(sample_num,data_path).to(device)

    conv_net = ConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad = pad,
        pad_mode=pad_mode,
        with_bias=WITH_BIAS,
        with_relu=with_relu,
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    inputs = dat[trace_id].detach()
    outputs = inputs.clone().detach()
    for sample_id in range(len(inputs)):
        image = inputs[sample_id].detach()
        save_image(os.path.join(save_path,f'output{sample_id}'),image,'input',is_norm=True,is_rgb=True)
        plot_fft(os.path.join(save_path,f'f_output{sample_id}'),image,'input',log_space=False)
    for layer_id in range(CONV_NUM):
        outputs = conv_net.main[layer_id](outputs)
        if (layer_id+1) % 5 == 0:
            for sample_id in range(len(inputs)):
                image = outputs[sample_id].detach()
                if layer_id == CONV_NUM - 1:
                    save_image(os.path.join(save_path,f'output{sample_id}'),image,f'layer{layer_id+1}',is_norm=True,is_rgb=True)
                    plot_fft(os.path.join(save_path,f'f_output{sample_id}'),image,f'layer{layer_id+1}',log_space=False)
                else:
                    save_image(os.path.join(save_path,f'output{sample_id}'),image,f'layer{layer_id+1}',is_norm=True,is_rgb=False)
                    plot_fft(os.path.join(save_path,f'f_output{sample_id}'),image,f'layer{layer_id+1}',log_space=False)


