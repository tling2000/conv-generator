from cProfile import label
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
from utils import get_logger, save_current_src,set_random,get_error,set_logger
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

def get_mean_low_freq_scale(image):
    assert len(image.shape) == 4
    N,C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    # f_image[:,:,0,0] = 0
    f_image = torch.fft.fftshift(f_image,dim = (-2,-1))
    f_image_power = torch.mean(torch.abs(f_image)**2,dim=(1))
    f_image_norm = f_image_power / torch.sum(f_image_power,dim=(-2,-1),keepdim=True)
    low_freq_scale = torch.sum(f_image_norm[:,int(H/2-H/16):int(H/2+H/16),int(W/2-W/16):int(W/2+H/16)],dim=(-2,-1))
    return low_freq_scale

if __name__ == '__main__':
    seed = 0
    device = 'cuda:0'
    sample_num = 100
    sample_net_num = 100
    with_relu = True
    
    save_root = f'/data2/tangling/conv-generator/outs/remark5/'
    
    data_paths = [
        '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt',
        '/data2/tangling/conv-generator/data/tiny-imagenet/image.pt',
        '/data2/tangling/conv-generator/data/broden1_224/image.pt',
    ]
    tags = [
        'cifar-10',
        'tiny-imagenet',
        'broden',
    ]
    kernel_sizes = [1,2,3,4,5,]

    
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')


    fig,ax = plt.subplots(figsize=(3,2))
    ax.set_xlabel('Kernel size K')
    ax.set_ylabel('Low frequency power ratio')
    ax.grid(True,linestyle='--')

    for i in range(len(data_paths)):
        set_random(seed)
        tag = tags[i]
        data_path = data_paths[i]
        
        dat = get_data(None,data_path)
        inputs = dat[:sample_num].detach().to(device)

        out_scale_lis = []

        for j in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[j]
            
            pad = kernel_size  // 2
            pad_mode = 'zeros'
            # pad = KERNEL_SIZE - 1
            # pad_mode = 'circular_one_side'

            conv_net = ConvNet(
                    kernel_size,
                    IN_CHANNELS,
                    MID_CHANNELS,
                    CONV_NUM,
                    pad = pad,
                    pad_mode=pad_mode,
                    with_bias=WITH_BIAS,
                    with_relu=with_relu,
                ).to(device)
            
            temp_lis = []
            for k in tqdm(range(sample_net_num)):
                conv_net.reset_params(PARAM_MEAN,PARAM_STD)

                outputs = conv_net(inputs)
                out_scale = get_mean_low_freq_scale(outputs).mean()
                temp_lis.append(out_scale)
            
            out_scale = np.array(temp_lis).mean()
            out_scale_lis.append(out_scale)

        ax.plot(kernel_sizes,out_scale_lis,marker='x',label=tag)

    ax.legend()
    fig.savefig(os.path.join(save_path,'ratio.jpg'),dpi=300,bbox_inches='tight')

