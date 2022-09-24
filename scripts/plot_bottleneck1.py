import os,sys
from turtle import forward
sys.path.append('../src')

import torch
from utils import save_image,get_fft,plot_sub_heatmap
from tqdm import tqdm




if __name__ == '__main__':
    rounds = 50
    # tag = 'cifar'
    data_path = '/data2/tangling/conv-generator/outs/bottleneck1/0924-202942-conv_num5-K3-in_channels3-mid_channels128-biasTrue'
    H,W = 64,64
    no_basis = False
    is_cut = False
    cut_scale = None
    insert_pixcel = 3

    inputs_dict = torch.load(os.path.join(data_path,'inputs_dic.pt'))
    outputs_dict = torch.load(os.path.join(data_path,'outputs_dic.pt'))

    for trace_id in inputs_dict.keys():
        print(trace_id)
        image = inputs_dict[trace_id]
        f_out = get_fft(image,no_basis=no_basis,is_cut=is_cut,cut_scale=cut_scale)
        save_image(os.path.join(data_path,'results'),image,f'sample{trace_id}_in',is_rgb=True)
        plot_sub_heatmap(os.path.join(data_path,'results'),[f_out],f'sample{trace_id}_inspec',cbar=False)

        outs,f_outs = [],[]
        outs.append(torch.ones((3,H,insert_pixcel)))
        for layer_id in [0,5,20,50,100,200,300,399]:
            image = outputs_dict[trace_id][layer_id]
            outs.append(image)
            outs.append(torch.ones((3,H,insert_pixcel)))

            f_out = get_fft(image,no_basis=no_basis,is_cut=is_cut,cut_scale=cut_scale)
            f_outs.append(f_out)

            out = torch.concat(outs,dim=2)
            save_image(os.path.join(data_path,'results_basis'),out,f'sample{trace_id}_out',is_rgb=True)
            plot_sub_heatmap(os.path.join(data_path,'results_basis'),f_outs,f'sample{trace_id}_outspec',cbar=False)
