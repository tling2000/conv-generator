import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_UPSAMPLE,WITH_BIAS,DATE,MOMENT
from models import ToyAE,ConvAE
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
from dat import get_gaussian_2d,get_data
from train import train1

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
    device = 'cuda:0'
    sample_num = 2000
    bs = 200
    lr = 0.0001
    rounds = 10
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/bottleneck2'

    # data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    # trace_ids = []
    # for i in range(0,1000,10):
    #     trace_ids.append(i)
    # insert_pixcel = 1

    data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/tiny-imagenet-200/sampled/image_64.pt'
    trace_ids = [120,1760,1340]
    H,W = 64,64
    # for i in range(0,2000,20):
    #     trace_ids.append(i)
    insert_pixcel = 3


    # data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    # insert_pixcel = 10
    # trace_ids = [16,17,26,31,35,40,46,59,72,79,80,82,90,98,391]
    # for i in range(500,1000,10):
    #     trace_ids.append(i)
    # trace_ids = [16,17,26,31,35,40,46,59,72,79,80,82,90,98,391,1328,1438,2393,2914,3035,4497,5600]


    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    conv_ae = ConvAE(
        'vgg16_bn',
        pretrained=True,
        image_shape=(H,W)
    )


    dat = get_data(sample_num,data_path)
    Xs = dat.detach()
    train1(
        conv_ae,
        Xs,
        save_path,
        rounds,
        batch_size=bs,
        lr=lr,
        device=device,
        trace_ids=trace_ids,
        insert_pixcel=insert_pixcel
    )

