import os,sys
from readline import insert_text
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, save_current_src,set_random,get_error,set_logger,save_image
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

def save_image(save_path,tensor,name,is_norm=False,is_rgb=False):
    assert len(tensor.shape) == 3,''
    tensor = tensor.detach().cpu()
    if is_rgb:
        pass
    else:
        tensor = tensor.mean(0)
    if is_norm:
        tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    unloader = transforms.ToPILImage()
    image = unloader(tensor)
    image.save(os.path.join(save_path,f'{name}.jpg'))

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


def get_fft(image,vmin=None,vmax=None):
    assert len(image.shape) == 3,''
    C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    f_image[:,0,0] = 0
    f_image = torch.fft.fftshift(f_image,dim=(-2,-1))
    f_image_norm = torch.abs(f_image).mean(0)
    
    # f_image_norm = f_image_norm[int(H/2-H/8):int(H/2+H/8),int(W/2-W/8):int(W/2+H/8)]
    f_image_norm = (f_image_norm-f_image_norm.min()) /(f_image_norm.max()-f_image_norm.min())
    return f_image_norm

if __name__ == '__main__':
    seed = 0
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
        with_relu=False,
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)
    relu = torch.nn.ReLU()

    dat = get_data(sample_num,data_path)
    inputs = dat[trace_id].detach().to(device)
    outputs = inputs.detach()

    out_list = []
    f_out_list = []
    for sample_id in range(len(inputs)):
        out_list.append([])
        f_out_list.append([])
        image = inputs[sample_id].detach()
        f_out = get_fft(image)
        out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
        out_list[sample_id].append(image)
        out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
        f_out_list[sample_id].append(f_out)
        
    for layer_id in range(CONV_NUM):
        outputs = conv_net.main[layer_id].conv(outputs)
        if layer_id in [1,3,9]:
            for sample_id in range(len(inputs)):
                image = outputs[sample_id].detach()
                out = torch.repeat_interleave(get_tensor(image).unsqueeze(0),repeats=3,dim=0)
                f_out = get_fft(image)
                out_list[sample_id].append(out)
                out_list[sample_id].append(torch.ones((3,H,insert_pixcel)))
                f_out_list[sample_id].append(f_out)
        outputs = relu(outputs)
    
    for sample_id in range(len(inputs)):
            
        out = torch.concat(out_list[sample_id],dim=2)
        save_image(save_path,out,f'out_{trace_id[sample_id]}',is_rgb=True)
        plot_heatmap(save_path,f_out_list[sample_id],f'f_out_{trace_id[sample_id]}',cbar=False)

