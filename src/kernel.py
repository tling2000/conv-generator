import os
import numpy as np
from utils import plot_heatmap,set_random
from tqdm import tqdm
import torch

def kernel_mask(
    kernel_size: int,
    image_shape: list,
    index: list,
    )->np.ndarray:
    H,W = image_shape
    u,v = index
    u_ = np.arange(H)
    v_ = np.arange(W)
    V,U = np.meshgrid(v_,u_)
    R = np.zeros((H,W),dtype=np.complex64)
    for t in range(kernel_size):
        for s in range(kernel_size):
            R += np.exp(1j*((U-u)*t/H + (V-v)*s/H)*2*np.pi)
    R = R /H/W
    return R

def kernel_fft(
    weights: torch.Tensor,
    img_shape: list,
    )->torch.Tensor:
    assert weights.shape[-2] == weights.shape[-1]
    assert len(weights.shape) == 4
    c_in,c_out,kernel_size = weights.shape[:-1]
    H,W = img_shape
    weights_expand = torch.zeros(size=(c_in,c_out,H,W),dtype=torch.float32)
    weights_expand[:,:,:kernel_size,:kernel_size] = weights
    weights_fft = torch.fft.ifft2(weights_expand) * (H*W)

    return weights_fft

def kernel_ifft(
    weights_fft: torch.Tensor,
    kernel_size: int,
    )->torch.Tensor:
    assert len(weights_fft.shape) == 4
    c_in,c_out,H,W = weights_fft.shape
    weights = torch.real(torch.fft.fft2(weights_fft)[:,:,:kernel_size,:kernel_size]) /(H*W)

    return weights

if __name__ == '__main__':
    save_path = '/data2/tangling/conv-expression/outs/lab7/Ruv'
    u,v = (5,10)
    H,W = (30,30)
    T = np.zeros((H,W))
    T[u,v] =  1
    plot_heatmap(save_path,T,'T')
    for i in [3,5,7]:
        K = i
        R = kernel_mask(K,(H,W),(u,v))
        plot_heatmap(save_path,np.abs(R),f'R_K{K}_sc',vmin=0,vmax=1)
        plot_heatmap(save_path,np.abs(R),f'R_K{K}')
    
    # weights = torch.zeros(4,4)
    # weights[:2,:2] = torch.randn(2,2)
    # weights = weights.requires_grad_()

    # weights_fft = torch.fft.ifft2(weights) * 4*4
    # iweights = torch.fft.fft2(weights_fft) / (4*4)
    # weights_fft.retain_grad()
    # iweights.retain_grad()

    # L = (iweights[:2,:2]).sum()
    # L.backward(retain_graph=True)

    # print(weights_fft.grad)
    # weights_fft += weights_fft.grad*0.1
    # print(torch.fft.fft2(weights_fft)/4/4)
    # print(weights + (weights.grad * 0.1 /4/4))
    # print(weights)

    # image = torch.randn(1,5,5).requires_grad_()
    # image_fft = torch.fft.fft2(image)
    # image_ = torch.fft.ifft2(image_fft)
    # image_fft.retain_grad()
    # image_.retain_grad()

    # L = (image_**2).sum()
    # L.backward(retain_graph=True)
    # print(torch.fft.ifft2(image_fft.grad)*5*5 - image_.grad)
    # print(image_fft.grad - (torch.fft.fft2(image_.grad)/5/5))

