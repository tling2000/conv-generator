import os
import numpy as np
from config import IMAGE_SHAPE, KERNEL_SIZE
from utils import plot_heatmap,set_random,get_error
from tqdm import tqdm
import torch


def alpha_trans_(
    kernel_size: int,
    image_shape: list,
    index: list,
    device: str
    )->torch.Tensor:
    H,W = image_shape
    H_,W_ = H-kernel_size+1,W-kernel_size+1
    u,v = index
    u_ = np.arange(H_)
    v_ = np.arange(W_)
    V,U = np.meshgrid(v_,u_)
    mask = np.zeros((H_,W_),dtype=np.complex64)
    for m in range(H_):
        for n in range(W_):
            mask += np.exp(1j*((u/H - U/H_)*m + ((v/W - V/W_)*n))*2*np.pi)
    mask = mask/H/W
    return torch.from_numpy(mask).to(device).detach()

def alpha_trans(
    kernel_size: int,
    image_shape: list,
    index: list,
    device: str
    )->torch.Tensor:
    H,W = image_shape
    K = kernel_size
    H_,W_ = H-kernel_size+1,W-kernel_size+1
    u,v = index
    u_ = np.arange(H_)
    v_ = np.arange(W_)
    V,U = np.meshgrid(v_,u_)

    lbd,gmm = (-u*(K-1)+(u-U)*H)/H/H_,(-v*(K-1)+(v-V)*W)/W/W_
    alphau = np.sin(lbd * np.pi * H_) / np.sin(lbd * np.pi) * np.exp(1j*(H-K)*lbd * np.pi)
    alphau[np.sin(lbd * np.pi)==0] = H_ 
    alphav = np.sin(gmm * np.pi * W_) / np.sin(gmm * np.pi) * np.exp(1j*(W-K)*gmm * np.pi)
    alphav[np.sin(gmm * np.pi)==0] = W_ 

    mask = alphau * alphav
    
    mask = mask / H/W

    return torch.from_numpy(mask).to(device).detach()

def delta_trans(
    kernel_size: int,
    image_shape: list,
    index: list,
    device: str
    )->torch.Tensor:
    H,W = image_shape
    u,v = index
    u_ = np.arange(H)
    v_ = np.arange(W)
    V,U = np.meshgrid(v_,u_)
    mask = np.zeros((H,W),dtype=np.complex64)
    for t in range(kernel_size):
        for s in range(kernel_size):
            mask += np.exp(1j*((U-u)*t/H + (V-v)*s/H)*2*np.pi)
    mask = mask/H/W
    return torch.from_numpy(mask).to(device).detach()

def Ruv(
    kernel_size: int,
    image_shape: list,
    device: str
    )->torch.Tensor:
    H,W = image_shape
    mask =  delta_trans(kernel_size,image_shape,(0,0),device)
    mask = mask * H * W
    return mask.detach()

def kernel_fft(
    weights: torch.Tensor,
    img_shape: list,
    device: str,
    )->torch.Tensor:
    assert weights.shape[-2] == weights.shape[-1]
    assert len(weights.shape) == 4
    c_out,c_in,kernel_size = weights.shape[:-1]
    H,W = img_shape
    weights_expand = torch.zeros(size=(c_out,c_in,H,W),dtype=torch.float32)
    weights_expand[:,:,:kernel_size,:kernel_size] = weights
    weights_fft = torch.fft.ifft2(weights_expand) * (H*W)

    return weights_fft.to(device).detach()

def kernel_ifft(
    weights_fft: torch.Tensor,
    kernel_size: int,
    device: str,
    )->torch.Tensor:
    assert len(weights_fft.shape) == 4
    c_in,c_out,H,W = weights_fft.shape
    weights = torch.real(torch.fft.fft2(weights_fft)[:,:,:kernel_size,:kernel_size]) /(H*W)

    return weights.to(device).detach()




if __name__ == '__main__':
    save_path = '/data2/tangling/conv-generator/outs/theorem1'




