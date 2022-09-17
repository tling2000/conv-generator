import os,sys
import re
sys.path.append('../src')

from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from utils import plot_heatmap
from dat import create_data

def plot_fft(save_path,image,name,log_space,vmin=None,vmax=None,cbar=False):
    assert len(image.shape) == 3,''
    C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    f_image[0,0] = 0
    f_image = torch.fft.fftshift(f_image,dim=(-2,-1))
    f_image_norm = torch.abs(f_image).mean(0)
    # f_image_power = torch.real(f_image * torch.conj(f_image))
    # f_image_norm = f_image_power / f_image_power.sum()
    if log_space:
        f_image_norm = torch.log10(f_image_norm)
        name = f'{name}_logspace'
    plot_heatmap(save_path,f_image_norm,name,vmin=vmin,vmax=vmax,cbar=cbar)

def save_image(save_path,tensor,name,is_norm=False,is_rgb=False):
    assert len(tensor.shape) == 3,''
    tensor = tensor.detach().cpu()
    if is_rgb:
        pass
    else:
        tensor = tensor.mean(0)
    if is_norm:
        tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    unloader = transforms.ToPILImage()
    image = unloader(tensor)
    image.save(os.path.join(save_path,f'{name}.jpg'))

def get_low_freq_scale(image):
    assert len(image.shape) == 3
    C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    f_image = torch.fft.fftshift(f_image,dim = (-2,-1))
    f_image_power = torch.real(f_image * torch.conj(f_image))
    f_image_norm = f_image_power / torch.sum(f_image_power,dim=(-2,-1),keepdim=True)
    low_freq_scale = torch.sum(f_image_norm[:,int(H/2-3):int(H/2+3),int(W/2-3):int(W/2+3)],dim=(-2,-1))
    return low_freq_scale

if __name__ == '__main__':
    data_path = '/data2/tangling/conv-generator/data/broden1_224/image.pt'
    save_path = '/data2/tangling/conv-generator/temp/fft-zero'

    images = torch.load(data_path)
    scale = get_low_freq_scale(images.mean(1))
    indexes = np.argsort(scale)[:100]
    
    for idx in indexes:
        plot_fft(save_path,images[idx],'f_image{}_{:.2f}'.format(idx,scale[idx]),log_space=False)
        # plot_fft(save_path,images[idx],f'f_image{idx}',log_space=True)
        save_image(save_path,images[idx],f'image{idx}',is_rgb=True)

