import torch
import numpy as np
import os
from config import MID_DIM,IMAGE_SHAPE,KERNEL_SIZE,CONV_NUM
from models import ConvNet
from kernel import kernel_fft,kernel_ifft,kernel_mask
from utils import plot_heatmap,set_random
from kernel import kernel_fft,kernel_ifft

if __name__ == '__main__':

    save_path = '/data2/tangling/conv-expression/outs/lab7/backward'

    layer_id = 4 #l-1

    seed = 0
    set_random(seed)
    conv_net = ConvNet(
        KERNEL_SIZE,
        MID_DIM,
        CONV_NUM,
    )
    input = torch.randn((1,MID_DIM,*IMAGE_SHAPE))
    output = conv_net(input)
    output.retain_grad()
    L = (output**2).sum()
    conv_net.zero_grad()
    L.backward()

    #path1
    in_image = input[0].detach() #C*H*W
    out_grad = output.grad[0].detach() 
    weights = conv_net.main[layer_id].conv.weight.detach()
    weights_grad = conv_net.main[layer_id].conv.weight.grad.detach()
    bias_grad = conv_net.main[layer_id].conv.bias.grad.detach()

    forward_fft = torch.conj(torch.fft.fft2(in_image)).detach() #C*H*W
    backward_fft = (torch.fft.fft2(out_grad)/ (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])).unsqueeze(0).detach() #1*C*H*W

    #forward
    for i in range(layer_id):
        weights = conv_net.main[i].conv.weight.detach()
        bias = conv_net.main[i].conv.bias.detach()
        bias_fft = bias * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
        weight_fft = kernel_fft(weights,IMAGE_SHAPE)

        for j in range(IMAGE_SHAPE[0]):
            for k in range(IMAGE_SHAPE[1]):
                if j == 0 and k == 0:
                    forward_fft[:,j,k] = torch.conj(weight_fft[:,:,j,k]) @ forward_fft[:,j,k] + bias_fft
                    print(bias_fft)
                else:
                    forward_fft[:,j,k] = torch.conj(weight_fft[:,:,j,k]) @ forward_fft[:,j,k]

    #backward
    for i in range(CONV_NUM-1,layer_id,-1):
        weights = conv_net.main[i].conv.weight.detach()
        weight_fft = kernel_fft(weights,IMAGE_SHAPE)
        for j in range(IMAGE_SHAPE[0]):
            for k in range(IMAGE_SHAPE[1]):
                backward_fft[:,:,j,k] = backward_fft[:,:,j,k] @ torch.conj(weight_fft[:,:,j,k])

    #real grad
    # real_grad = kernel_fft(weights_grad,IMAGE_SHAPE) / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    # real_grad =  torch.transpose(real_grad,0,1)

    #calucate grad
    grad_uncons = torch.zeros(MID_DIM,MID_DIM,*IMAGE_SHAPE,dtype=torch.complex64)
    grad_cons = torch.zeros(MID_DIM,MID_DIM,*IMAGE_SHAPE,dtype=torch.complex64)
    for j in range(IMAGE_SHAPE[0]):
        for k in range(IMAGE_SHAPE[1]):
            grad_uncons[:,:,j,k] = forward_fft[:,j,k].unsqueeze(1) @ backward_fft[:,:,j,k]

    for u in range(IMAGE_SHAPE[0]):
        for v in range(IMAGE_SHAPE[1]):
            beta_uv = torch.torch.from_numpy(kernel_mask(KERNEL_SIZE,IMAGE_SHAPE,[u,v]))
            grad_cons += grad_uncons[:,:,u,v].unsqueeze(-1).unsqueeze(-1) * beta_uv.unsqueeze(0).unsqueeze(0)

    #verify
    weight_fft = kernel_fft(weights,IMAGE_SHAPE)
    weight_fft += torch.transpose(grad_cons,0,1)
    iweights = kernel_ifft(weight_fft,KERNEL_SIZE)
    delta_weight = (iweights - weights)*(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    delta_weight_real = weights_grad

    delta = (delta_weight - delta_weight_real)
    delta_norm = np.linalg.norm(delta.numpy().reshape(-1),ord=2)
    grad_norm = np.linalg.norm(delta_weight_real.numpy().reshape(-1),ord=2)

    print(delta_norm/grad_norm)

    #compare
    # delta = (real_grad - grad_cons)
    # delta_norm = np.linalg.norm(delta.numpy().reshape(-1),ord=2)
    # grad_norm = np.linalg.norm(real_grad.numpy().reshape(-1),ord=2)

    # print(delta_norm/grad_norm)