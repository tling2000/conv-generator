from email.mime import image
from torchvision import transforms
import PIL
import torch
from utils import plot_heatmap
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import stats

if __name__ == '__main__':
    data_path = '/data2/tangling/conv-expression/data/lab4/image.pt'
    save_path = '/data2/tangling/conv-expression/outs/lab7/insert/imagenet'
    reszie_shape = (96,96)

    images = torch.load(data_path)
    transform = transforms.Compose([
        transforms.RandomCrop(48),
        transforms.Resize((64,64))
    ])
    
    
    images_insert = transform(images)

    images_fft = torch.fft.fftshift(torch.fft.fft2(images),(-2,-1))
    images_insert_fft =  torch.fft.fftshift(torch.fft.fft2(images_insert),(-2,-1))

    images_fft_mean = torch.log(torch.mean(torch.abs(images_fft),dim=(0,1))**2)
    images_in_fft_mean = torch.log(torch.mean(torch.abs(images_insert_fft),dim=(0,1))**2)

    u = np.arange(-32,32)
    v = np.arange(-32,32)
    U,V = np.meshgrid(u,v)
    k = np.log((U**2 + V**2) ** 0.5)

    fig,ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.plot_surface(U,V,images_fft_mean.numpy(),cmap='jet')
    ax.set_zlim(-2,11)
    fig.savefig(os.path.join(save_path,f'image_power.png'))

    fig,ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.plot_surface(U,V,images_in_fft_mean.numpy(),cmap='jet')
    ax.set_zlim(-2,11)
    fig.savefig(os.path.join(save_path,f'image_in_power.png'))

    k[32,32] = k[31,31]
    images_fft_mean[32,32] = images_fft_mean[31,31]
    images_fft_mean[32,32] = images_fft_mean[31,31]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.scatter(k.reshape(-1),images_fft_mean.numpy().reshape(-1),s=4)
    ax.grid()
    ax.set_ylim(-2,11)
    fig.savefig(os.path.join(save_path,'image_power_line.png'))
    images_in_fft_mean[32,32] = images_in_fft_mean[31,31]
    images_in_fft_mean[32,32] = images_in_fft_mean[31,31]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.scatter(k.reshape(-1),images_in_fft_mean.numpy().reshape(-1),s=4)
    ax.grid()
    ax.set_ylim(-2,11)
    fig.savefig(os.path.join(save_path,'image_in_power_line.png'))

    for i in range(0):
        plot_heatmap(save_path,images[i,0],f'image_{i}')
        plot_heatmap(save_path,images_insert[i,0],f'image_in_{i}')
        plot_heatmap(save_path,torch.abs(images_fft[i,0]),f'image_fft_{i}',vmax=300,vmin=0)
        plot_heatmap(save_path,torch.abs(images_insert_fft[i,0]),f'image_in_fft_{i}',vmax=300,vmin=0)

    plot_heatmap(save_path,torch.abs(images_fft_mean),f'image_fft_mean',vmax=300,vmin=0)
    plot_heatmap(save_path,torch.abs(images_in_fft_mean),f'image_in_fft_mean',vmax=300,vmin=0)