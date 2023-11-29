import os,sys
sys.path.append('../src')

import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from utils import plot_heatmap,save_image

if __name__ == '__main__':
    # load_path = '/data2/tangling/conv-generator/outs/exp1/0106-114940'
    # save_path = f'{load_path}/exp1/samples'

    load_path = '/data2/tangling/conv-generator/outs/exp2/0106-114943'
    save_path = f'{load_path}/exp2/samples'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    in_image = torch.load(os.path.join(load_path,'in_image.pt')).detach()
    out_image = torch.load(os.path.join(load_path,f'out_image_400.pt')).detach()
    in_fft = torch.fft.fft2(in_image).detach()
    in_fft[:,:,0,0] = 0
    in_fft = torch.fft.fftshift(in_fft,dim=(-2,-1))
    out_fft = torch.fft.fft2(out_image).detach()
    out_fft[:,:,0,0] = 0
    out_fft = torch.fft.fftshift(out_fft,dim=(-2,-1))

    for i in range(0,2000,200):
        save_image(save_path,in_image[i],f'in_image_{i}',is_norm=True,is_rgb=True)
        save_image(save_path,out_image[i],f'out_image_{i}',is_norm=True,is_rgb=True)

        plot_heatmap(save_path,torch.abs(torch.mean(in_fft[i],dim=0)),f'in_fft_{i}')
        plot_heatmap(save_path,torch.abs(torch.mean(out_fft[i],dim=0)),f'out_fft_{i}')
        