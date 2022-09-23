import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import ConvNet
from coef import kernel_fft,alpha_trans
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger,get_cos
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

    seed = 1000
    device = 'cuda:1'
    sample_num = 50
    pad = KERNEL_SIZE  // 2
    pad_mode = 'zeros'

    is_transform = False
    
    # with_relu = False
    with_relu = True

    
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = '/data2/tangling/conv-generator/outs/corollary1'

    # data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    # data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/image.pt'
    data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'

    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    # init the conv net
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

    inputs = get_data(sample_num,data_path).to(device) #1*C*H*W
    if is_transform:
        transform = transforms.Resize(IMAGE_SHAPE)
        inputs = transform(inputs)

    cos_lis = []
    for sample_id in tqdm(range(sample_num)):
        #sample the input
        cos_lis.append([])

        input = inputs[sample_id:sample_id+1]
        f_input = torch.fft.fft2(input)[0].detach() #C*H*W
        output = input.detach()

        for layer_id in range(CONV_NUM):
            output = conv_net.main[layer_id](output) #1*C*H*W
            f_output = torch.fft.fft2(output)[0].detach() #C*H*W
            #if relu
            if with_relu:
                output = relu(output)

            #cal
            T,beta = conv_net.get_freq_trans(IMAGE_SHAPE,(0,layer_id+1),device)
            if layer_id == CONV_NUM-1:
                cal_f_output = torch.zeros((IN_CHANNELS,H,W),dtype=torch.complex64,device=device) #C*H*W
            else:
                cal_f_output = torch.zeros((MID_CHANNELS,H,W),dtype=torch.complex64,device=device) #C*H*W
            #mat multiply
            for u in range(H):
                for v in range(W):
                    cal_f_output[:,u,v] = T[:,:,u,v] @ f_input[:,u,v]
            #add the basic frequency
            # cal_f_output[:,0,0] += beta

            #get error
            cos = get_cos(f_output.cpu().numpy(),cal_f_output.cpu().numpy(),dims=(-2,-1))
            cos_lis[sample_id].append(cos)

    mean_cos = np.array(cos_lis).mean(0)
    std_cos = np.array(cos_lis).std(0)
    np.save(os.path.join(save_path,'mean_cos.npy'),mean_cos)
    np.save(os.path.join(save_path,'std_cos.npy'),std_cos)
    logger.info(f'mean cos:{mean_cos}')
    logger.info(f'std cos:{std_cos}')


