import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import math

indexes_0 = [(0,1),(0,-1),(1,0),(-1,0)]
indexes_1 = [(-1,1),(-1,-1),(1,-1),(1,1)]
indexes_2 = [(-1,-2),(-1,2),(1,-2),(1,2),(-2,-1),(-2,1),(2,1),(2,-1)]
indexes_3 = [(-2,2),(-2,-2),(2,-2),(2,2)]
indexes_4 = [(-3,-2),(-3,2),(3,-2),(3,2),(-2,-3),(-2,3),(2,3),(2,-3)]
indexes_list = [indexes_0,indexes_1,indexes_2,indexes_3,indexes_4]
len_list = [1,math.sqrt(2),math.sqrt(5),math.sqrt(8),math.sqrt(13)]

def is_leagal(index,shape):
    H,W = shape
    x,y = index
    if x<0 or x>=H:
        return False
    if y<0 or y>=W:
        return False
    return True

def cal_cost(fft,indexes,shape):
    count = 0
    value = 0
    for i in range(H):
        for j in range(W):
            for nbr in indexes:
                index = np.array((i,j))
                new_index = index + np.array(nbr)
                if not is_leagal(new_index,shape):
                    continue
                count += 1
                value += torch.abs(fft[:,:,index[0],index[1]] - fft[:,:,new_index[0],new_index[1]]).mean() / 2
    cost = value / count
    return cost.detach()


if __name__ == '__main__':
    load_path = '/data2/tangling/conv-generator/outs/exp1/0106-114940'
    save_path = '/data2/tangling/conv-generator/outs/exp1/0106-114940/exp1'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig1,ax1 = plt.subplots()
    fig2,ax2 = plt.subplots()
    in_image = torch.load(os.path.join(load_path,'in_image.pt'))
    H,W = in_image.shape[-2:]
    in_fft = torch.fft.fft2(in_image)
    in_costs = []
    in_rela_costs = []
    for i in range(len(indexes_list)):
        indexes = indexes_list[i]
        cost = cal_cost(in_fft,indexes,(H,W)).item()
        rela_cost = cost / torch.abs(in_fft).mean().item()
        print(torch.abs(in_fft).mean().item())
        in_costs.append(cost)
        in_rela_costs.append(rela_cost)
    ax1.plot(len_list,in_costs,label='in_cost')
    ax2.plot(len_list,in_rela_costs,label='in_re_cost')

    for rnd in [0,50,200,400]:
        out_image = torch.load(os.path.join(load_path,f'out_image_{rnd}.pt'))
        out_fft = torch.fft.fft2(out_image)
        out_costs = []
        out_rela_costs = []
        for i in range(len(indexes_list)):
            indexes = indexes_list[i]
            cost = cal_cost(out_fft,indexes,(H,W)).item()
            rela_cost = cost / torch.abs(out_fft).mean().item()
            print(rnd)
            print(torch.abs(out_fft).mean().item())
            out_costs.append(cost)
            out_rela_costs.append(rela_cost)
        ax1.plot(len_list,out_costs,label=f'out_cost_rnd{rnd}')
        ax2.plot(len_list,out_rela_costs,label=f'in_re_cost_rnd{rnd}')

    ax1.legend()
    ax1.grid()
    fig1.savefig(os.path.join(save_path,'cost.jpg'))

    ax2.legend()
    ax2.grid()
    fig2.savefig(os.path.join(save_path,'rela_cost.jpg'))