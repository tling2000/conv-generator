from email.mime import base
import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
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

def plot_mean_abs(save_path,mat,name):
    assert len(mat.shape) == 4,''
    mean_abs = torch.mean(torch.abs(mat),dim=(0,1)).detach()
    mean_abs_norm = mean_abs / mean_abs.max()
    plot_heatmap(save_path,mean_abs_norm,name,cbar=False)

if __name__ == '__main__':
    seed = 0
    device = 'cuda:1'
    sample_num = 50
    sample_net_num = 100
    with_relu = True

    pad = KERNEL_SIZE  // 2
    pad_mode = 'zeros'

    # pad = KERNEL_SIZE - 1
    # pad_mode = 'circular_one_side'

    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = f'/data2/tangling/conv-generator/outs/remark1/'

    tag = 'cifar'
    data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    # trace_index = (0,1,2,4,8,16)
    trace_index = ((1,10),(3,7),(5,2),(8,4),(9,5))

    # tag = 'imagenet'
    # data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/image.pt'
    # trace_index = (0,1,2,4,8,16,32)

    # tag = 'broden'
    # data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    # trace_index = (0,1,2,4,8,16,32,64,112)

    
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
        with_relu=False,
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)
    relu = torch.nn.ReLU()

    # get som lis
    # soms_lis = []
    # for j in  tqdm(range(sample_net_num)):
    #     som_lis = []
    #     outputs = dat.detach()
    #     conv_net.reset_params(PARAM_MEAN,PARAM_STD)
    #     for i in range(CONV_NUM):
    #         outputs = conv_net.main[i].conv(outputs)
    #         f_outputs = torch.fft.fft2(outputs.detach().cpu())
    #         som = torch.real(f_outputs * torch.conj(f_outputs))
    #         som_lis.append(som[:,0].mean(0,keepdim=True))

    #         if with_relu:
    #             outputs = relu(outputs)
    #     som_array = torch.concat(som_lis)
    #     soms_lis.append(som_array.unsqueeze(0))
    # soms_array = torch.concat(soms_lis).mean(0)
    # torch.save(soms_array,os.path.join(save_path,'soms.pt'))

    soms_array = torch.load('/data2/tangling/conv-generator/outs/remark1/0924/0924-210257-conv_num50-K5-in_channels3-mid_channels16-biasFalse-mean0-std0.1/soms.pt')
    
    # colors = ['#FDE725FF','#DCE319FF','#B8DE29FF','#95D840FF','#73D055FF','#55C667FF','#3CBB75FF','#29AF7FFF','#20A387FF']
    colors = ['#ff432e','#e64f43','#cd5b58','#b4676c','#9b7381','#838096','#6a8cab','#5198c0','#38a4d4',' #1fb0e9',' #1fb0e9','#06bcfe']
    # colors = plt.get_cmap('Set2')
    #plot
    x = range(1,CONV_NUM+1)
    fig,ax = plt.subplots(figsize=(4.2,2.8))
    # ax.set_yscale('log',base=10)
    ax.grid(True, which = 'both',linestyle='--')
    ax.set_ylabel(r'$\mathbf{log}$ SOM($\mathbf{h}^{(uv)}$)',fontdict={'size':14})
    ax.set_xlabel('Depth of the network',fontdict={'size':14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    lenth = len(trace_index)
    for i,index in enumerate(trace_index):
        # ax.plot(x,torch.log10(soms_array[:,index,index]),label=f'u=v={index}',c='red',alpha=1-(i/(len(trace_index)+2)))
        # ax.plot(x,torch.log10(soms_array[:,index[0],index[1]]),label=f'u={index[0]},v={index[1]}',c = 1-(i/(len(trace_index)+2)),cmap='viridis')

        # ax.plot(x,torch.log10(soms_array[:,index,index]),label=f'u=v={index}',c=colors[i*9//lenth])
        ax.plot(x,torch.log10(soms_array[:,index[0],index[1]]),label=f'u={index[0]},v={index[1]}',c=colors[i*9//lenth])

    ax.legend(loc=2,ncol=1,fontsize=13)
    fig.savefig(os.path.join(save_path,f'{tag}_som.pdf'),bbox_inches='tight')
    