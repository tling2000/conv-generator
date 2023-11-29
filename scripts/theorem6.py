import os,sys
sys.path.append('../src')

import torch
from utils import get_logger, save_current_src,set_random,set_logger,save_image,get_fft,plot_sub_heatmap
from dat import get_data
from tqdm import tqdm

def get_tensor(tensor,):
    tensor = tensor.mean(0)
    tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    return tensor

def make_dirs(save_root,tag):
    save_path = os.path.join(save_root,tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def upsample(images,ratio):
    C,H,W = images.shape
    up_images = torch.zeros(C,H*ratio,W*ratio)
    up_images[:,::ratio,::ratio] = images
    return up_images.detach()

if __name__ == '__main__':
    seed = 0

    save_root = f'/data2/tangling/conv-generator/outs/theorem6/'

    # tag = 'cifar'
    # data_path = '/data2/tangling/conv-generator/data/cifar-10-batches-py/image.pt'
    # trace_id = range(0,100)
    # insert_pixcel = 1
    # is_cut = False
    # cut_scale = None
    # H,W = 32,32

    # tag = 'imagenet'
    # data_path = '/data2/tangling/conv-generator/data/tiny-imagenet/tiny-imagenet-200/sampled2/image_64.pt'
    # trace_id = range(0,2000,50)
    # insert_pixcel = 3
    # is_cut = False
    # cut_scale = None
    # H,W = 64,64

    tag = 'broden'
    data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    trace_id = [16,17,26,31,35,40,46,59,72,79,80,82,90,98,391,1328,1438,2393,2914,3035,4497,5600]
    insert_pixcel = 10
    is_cut = True
    cut_scale = 4
    H,W = 224,224
    
    save_path = make_dirs(save_root,tag)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    dat = get_data(None,data_path)
    inputs = dat[trace_id]
    print(inputs.shape)
    max_ratio = 16

    for sample_id in tqdm(range(len(inputs))):
        image = inputs[sample_id].detach()
        f_out = get_fft(image,no_basis=False,is_cut=is_cut,cut_scale=cut_scale)
        save_image(save_path,image,f'sample{trace_id[sample_id]}_in',is_rgb=True)
        plot_sub_heatmap(save_path,[f_out],f'sample{trace_id[sample_id]}_inspec',cbar=False)

        # out_list = []
        f_out_list = []
        # out_list.append(torch.ones((3,H*max_ratio,insert_pixcel*max_ratio)))
        for ratio in [2,4,8,max_ratio]:
            image = upsample(inputs[sample_id],ratio)
            f_out = get_fft(image,no_basis=False,is_cut=is_cut,cut_scale=cut_scale)

            # image_temp = torch.zeros(3,H* max_ratio,W* max_ratio)
            # scale = H * max_ratio // image.shape[1]
            # for i in range(scale):
            #     for j in range(scale):
            #         image_temp[:,i::scale,j::scale] = image

            # out_list.append(image_temp)
            # out_list.append(torch.ones((3,H*max_ratio,insert_pixcel*max_ratio)))
            f_out_list.append(f_out)
    
        # out = torch.concat(out_list,dim=2)
        # save_image(save_path,out,f'sample{trace_id[sample_id]}_out',is_rgb=True)
        plot_sub_heatmap(save_path,f_out_list,f'sample{trace_id[sample_id]}_outspec',cbar=False)


