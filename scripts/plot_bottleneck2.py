import os,sys
from turtle import forward
sys.path.append('../src')

import torch
from utils import save_image,get_fft,plot_sub_heatmap
from tqdm import tqdm




if __name__ == '__main__':
    rounds = 50
    # tag = 'cifar'
    data_path = '/data2/tangling/conv-generator/outs/bottleneck2/cifar'
    H,W = 32,32
    insert_pixcel = 1

    #tag = 'tiny-imagenet'
    # data_path = '/data2/tangling/conv-generator/outs/bottleneck2/imagenet'
    # H,W = 64,64
    # insert_pixcel = 3

    # 'broden'
    # data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    # insert_pixcel = 10

    inputs_dict = torch.load(os.path.join(data_path,'inputs_dict.pt'))
    outputs_dict = torch.load(os.path.join(data_path,'outputs_dict.pt'))

    for epoch in tqdm(range(rounds)):
        for trace_id in inputs_dict.keys():
            outs = []
            f_outs = []
            f_outs_nozero = []

            image = inputs_dict[trace_id]
            outs.append(torch.ones((3,H,insert_pixcel)))
            outs.append(image)
            outs.append(torch.ones((3,H,insert_pixcel)))
            
            f_out = get_fft(image,no_basis=False,is_cut=False)
            f_out_nozero = get_fft(image,no_basis=True,is_cut=False)
            f_outs.append(f_out)
            f_outs_nozero.append(f_out_nozero)
            
            for layer_id in outputs_dict[trace_id].keys():
                
                image = torch.from_numpy(outputs_dict[trace_id][layer_id][epoch])
                image = torch.repeat_interleave(get_tensor(image).unsqueeze(0),repeats=3,dim=0)
                image_temp = torch.zeros(3,H,W)
                scale = H // image.shape[1]
                for i in range(scale):
                    for j in range(scale):
                        image_temp[:,i::scale,j::scale] = image

                outs.append(image_temp)
                outs.append(torch.ones((3,H,insert_pixcel)))

                f_out = get_fft(image,no_basis=False,is_cut=False)
                f_out_nozero = get_fft(image,no_basis=True,is_cut=False)
                f_outs.append(f_out)
                f_outs_nozero.append(f_out_nozero)

            out = torch.concat(outs,dim=2)
            save_image(os.path.join(data_path,'image',f'epoch{epoch}'),out,f'{trace_id}',is_rgb=True)

            plot_sub_heatmap(os.path.join(data_path,'image',f'epoch{epoch}'),f_outs,f'f_{trace_id}',cbar=False)
            plot_sub_heatmap(os.path.join(data_path,'image',f'epoch{epoch}'),f_outs_nozero,f'f_{trace_id}_nozero',cbar=False)

