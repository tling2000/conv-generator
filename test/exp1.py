import os,sys
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm

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

if __name__ == '__main__':
    seed = 0
    device = 'cuda:1'
    sample_num = 2000
    lr = 0.001
    rounds = 400

    save_root = f'/data2/tangling/conv-generator/outs/exp1'
    data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/image.pt'


    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../test')
    
    toyae = ToyConvAE(
        3,
        3,
        16,
        5,
        )

    dat = get_data(sample_num,data_path)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(toyae.parameters(),lr=lr)

    loss_list = []
    for i in tqdm(range(rounds+1)):
        in_image = dat.detach()
        out_image = toyae(in_image)

        optimizer.zero_grad()
        loss = loss_func(in_image,out_image)
        loss.backward()
        optimizer.step()
        logger.info(f'rnd: {i}, loss: {loss}')

        loss_list.append(loss.item())
        if i % 50 == 0 or i == rounds:
            torch.save(out_image,os.path.join(save_path,f'out_image_{i}.pt'))
            torch.save(toyae.state_dict(),os.path.join(save_path,'in_image.pt'))

    torch.save(in_image,os.path.join(save_path,'in_image.pt'))
    

    fig,ax = plt.subplots()
    ax.plot(loss_list)
    fig.savefig(os.path.join(save_path,'loss.png'))

    np.save(os.path.join(save_path,'loss.npy'),np.array(loss_list))
