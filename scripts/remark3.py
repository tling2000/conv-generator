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
from utils import get_logger, save_current_src,set_random,get_error,set_logger,get_fft,save_image
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

def get_tensor(tensor,):
    tensor = tensor.mean(0)
    tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    return tensor


def plot_heatmap(save_path: str, 
                 mats: list, 
                 name: str, 
                 vmin: int = None, 
                 vmax: int = None,
                 cbar: bool = True,
                 ) -> None: 
    assert (vmin is None) == (vmax is None), "vmin and vmax must be both None or not None"

    feature_num = len(mats)
    fig, ax = plt.subplots(1,feature_num,figsize=(feature_num*4,4))
    for i in range(feature_num):
        sns.heatmap(mats[i], annot=False, cbar=cbar, cmap = 'coolwarm', vmin = vmin, vmax = vmax,ax=ax[i]) 
        ax[i].set_axis_off()  
    # ax.set_title(name) 
    # plt.axis('off')
    fig.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if vmin is None:
        path = os.path.join(save_path, f"{name}.png")
    else:
        path = os.path.join(save_path, "{}_{:.4f}to{:.4f}.png".format(name, vmin, vmax)) 
    fig.savefig(path,dpi=300) 
    plt.close()

if __name__ == '__main__':
    seed = 1
    device = 'cuda:0'
    sample_num = None
    with_relu = True

    pad = KERNEL_SIZE  // 2
    pad_mode = 'zeros'

    # pad = KERNEL_SIZE - 1
    # pad_mode = 'circular_one_side'

    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE
    
    save_root = f'/data2/tangling/conv-generator/outs/remark3/'
    
    data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    trace_id = range(100)
    insert_pixcel = 1

    # data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/image.pt'
    # trace_id = [11,17,28,39,47,68,69,73,79,81]
    # insert_pixcel = 3

    # data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    # trace_id = [16,17,26,31,35,40,46,59,72,79,80,82,90,98,391,1328,1438,2393,2914,3035,4497,5600]
    # insert_pixcel = 10

    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')
    
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

    dat = get_data(sample_num,data_path)
    inputs = dat[trace_id].detach().to(device)

    out_list = []
    f_out_list = []
    for sample_id in range(len(inputs)):
        out_list.append([])
        f_out_list.append([])
        image = inputs[sample_id].detach().cpu()
        f_out = get_fft(image)
        out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
        out_list[sample_id].append(image)
        out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
        f_out_list[sample_id].append(f_out)

    for param_mean in [0,0.001,0.01]:

        conv_net.reset_params(param_mean,PARAM_STD)

        outputs = conv_net(inputs)
        for sample_id in range(len(inputs)):
            image = outputs[sample_id].detach().cpu()
            out = torch.repeat_interleave(get_tensor(image).unsqueeze(0),repeats=3,dim=0)
            f_out = get_fft(image)
            out_list[sample_id].append(out)
            out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
            f_out_list[sample_id].append(f_out)

    for sample_id in range(len(inputs)):
        out = torch.concat(out_list[sample_id],dim=2)
        save_image(save_path,out,f'out_{trace_id[sample_id]}',is_rgb=True)
        plot_heatmap(save_path,f_out_list[sample_id],f'f_out_{trace_id[sample_id]}',cbar=False)