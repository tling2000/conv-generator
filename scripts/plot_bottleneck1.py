import os,sys
from turtle import forward
sys.path.append('../src')

import torch
from utils import save_image,get_fft,plot_sub_heatmap
from tqdm import tqdm




if __name__ == '__main__':
    data_path = '/data2/tangling/conv-generator/outs/bottleneck1/0924/broden'
    H,W = 224,224
    insert_pixcel = 10

    no_basis = True
    is_cut = True
    cut_scale = 4
    tag = 'tail'

    inputs_dict = torch.load(os.path.join(data_path,'inputs_dic.pt'))
    outputs_dict = torch.load(os.path.join(data_path,'outputs_dic.pt'))

    for trace_id in inputs_dict.keys():
        print(trace_id)
        image = inputs_dict[trace_id]
        f_out = get_fft(image,no_basis=no_basis,is_cut=is_cut,cut_scale=cut_scale)
        save_image(os.path.join(data_path,f'{tag}'),image,f'sample{trace_id}_in',is_rgb=True)
        plot_sub_heatmap(os.path.join(data_path,f'{tag}'),[f_out],f'sample{trace_id}_inspec',cbar=False)

        outs,f_outs = [],[]
        outs.append(torch.ones((3,H,insert_pixcel)))
        for layer_id in [50,100,200,300,399]:
            image = outputs_dict[trace_id][layer_id]
            outs.append(image)
            outs.append(torch.ones((3,H,insert_pixcel)))

            f_out = get_fft(image,no_basis=no_basis,is_cut=is_cut,cut_scale=cut_scale)
            f_outs.append(f_out)

            out = torch.concat(outs,dim=2)
            save_image(os.path.join(data_path,f'{tag}'),out,f'sample{trace_id}_out',is_rgb=True)
            plot_sub_heatmap(os.path.join(data_path,f'{tag}'),f_outs,f'sample{trace_id}_outspec',cbar=False)
