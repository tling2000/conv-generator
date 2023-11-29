import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from models import ToyConvAE

from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger
from dat import get_gaussian_2d,get_data

from matplotlib import pyplot as plt

from config import DATE,MOMENT

def make_dirs(save_root):
    exp_name = "-".join([DATE, MOMENT,])
    save_path = os.path.join(save_root, exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def resize(images,size):
    H,W = images.size()[-2:]
    transform = transforms.Resize(size)
    new_images = torch.zeros_like(images)
    start_h = (H- size[0] ) // 2
    start_w = (W- size[1] ) // 2
    new_images[:,:,start_h:start_h + size[0],start_w:start_w+size[1]] = transform(images)
    return new_images.detach()


if __name__ == '__main__':
    seed = 0
    device = 'cuda:0'
    sample_num = 1000
    lr = 0.001
    rounds = 400
    tgt_size = (63,63)

    save_root = f'/data2/tangling/conv-generator/outs/exp2/0108'
    data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/image_s1000.npy'

    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../test')
    
    toyae = ToyConvAE(
        kernel_size = 3,
        in_channels = 3,
        mid_channels = 16,
        conv_num = 5,
        )

    dat = get_data(sample_num,data_path)
    in_image = dat.detach()
    target = resize(in_image,tgt_size)

    H,W = in_image.size()[-2:]
    s_h = (H - tgt_size[0]) // 2
    s_w = (W - tgt_size[1]) // 2

    torch.save(in_image.detach(),os.path.join(save_path,'in_image.pt'))
    torch.save(target.detach(),os.path.join(save_path,'target.pt'))
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(toyae.parameters(),lr=lr)

    toyae.to(device)
    loss_func.to(device)
    in_image = in_image.to(device)
    target = target.to(device)

    loss_list = []
    for i in tqdm(range(rounds+1)):
        out_image = toyae(in_image)
        optimizer.zero_grad()
        loss = loss_func(out_image[:,:,s_h:s_h + tgt_size[0],s_w:s_w + tgt_size[1]],target[:,:,s_h:s_h + tgt_size[0],s_w:s_w + tgt_size[1]])
        loss.backward()
        optimizer.step()
        logger.info(f'rnd: {i}, loss: {loss}')

        loss_list.append(loss.item())
        if i % 100 == 0 or i == rounds:
            torch.save(out_image.detach(),os.path.join(save_path,f'out_image_{i}.pt'))
            torch.save(toyae.state_dict(),os.path.join(save_path,f'model_{i}.pt'))

    fig,ax = plt.subplots()
    ax.plot(loss_list)
    fig.savefig(os.path.join(save_path,'loss.png'))

    np.save(os.path.join(save_path,'loss.npy'),np.array(loss_list))
