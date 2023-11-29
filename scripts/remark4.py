import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
import seaborn as sns

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet
from utils import get_logger, save_current_src,set_random,get_error,set_logger,plot_sub_heatmap,get_fft,get_tensor,save_image
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
    seed = 1
    device = 'cuda:0'
    sample_num = None
    with_relu = True

    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE
    
    save_root = f'/data2/tangling/conv-generator/outs/remark4/'
    
    # data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    # trace_id = range(100)
    # insert_pixcel = 1
    # is_cut = False
    # cut_scale = None
    # H,W = 32,32

    data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/tiny-imagenet-200/sampled2/image_64.pt'
    trace_id = range(0,2000,40)
    insert_pixcel = 3
    is_cut = False
    cut_scale = None
    H,W = 64,64

    # data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    # trace_id = [16,17,26,31,35,40,46,59,72,79,80,82,90,98,391,1328,1438,2393,2914,3035,4497,5600]
    # insert_pixcel = 10
    # is_cut = True
    # cut_scale = 4
    # H,W = 224,224

    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')
    

    dat = get_data(sample_num,data_path)
    inputs = dat[trace_id].detach().to(device)

    out_list = []
    f_out_list = []
    for sample_id in range(len(inputs)):
        out_list.append([])
        f_out_list.append([])
        out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))

        image = inputs[sample_id].detach()
        f_out = get_fft(image,no_basis=True,is_cut=is_cut,cut_scale=cut_scale)
        save_image(save_path,image,f'sample{trace_id[sample_id]}_in',is_rgb=True)
        plot_sub_heatmap(save_path,[f_out],f'sample{trace_id[sample_id]}_inspec',cbar=False)

    for i in range(3):
        pad_mode = ['zeros','circular_one_side','reflect'][i]
        pad = [KERNEL_SIZE  // 2,KERNEL_SIZE-1, KERNEL_SIZE  // 2][i]

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

        outputs = conv_net(inputs)
        for sample_id in range(len(inputs)):
            image = outputs[sample_id].detach().cpu()
            out = torch.repeat_interleave(get_tensor(image).unsqueeze(0),repeats=3,dim=0)
            f_out = get_fft(image,no_basis=True,is_cut=is_cut,cut_scale=cut_scale)
            out_list[sample_id].append(out)
            out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
            f_out_list[sample_id].append(f_out)

    for sample_id in range(len(inputs)):
        out = torch.concat(out_list[sample_id],dim=2)
        save_image(save_path,out,f'sample{trace_id[sample_id]}_out',is_rgb=True)
        plot_sub_heatmap(save_path,f_out_list[sample_id],f'sample{trace_id[sample_id]}_outspec',cbar=False)