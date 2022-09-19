import os,sys
from turtle import forward
sys.path.append('../src')

import torch
import numpy as np
from tqdm import tqdm

from config import CONV_NUM, KERNEL_SIZE,IMAGE_SHAPE,IN_CHANNELS,MID_CHANNELS,WITH_BIAS,PARAM_MEAN,PARAM_STD,DATE,MOMENT
from models import CircuConvNet
from coef import kernel_fft,delta_trans,kernel_ifft
from utils import get_logger, plot_heatmap, save_current_src,set_random,get_error,set_logger,get_cos
from dat import get_data

def make_dirs(save_root):
    exp_name = "-".join([DATE, 
                        MOMENT,
                        f"conv_num{CONV_NUM}",
                        f"K{KERNEL_SIZE}",
                        f"in_channels{IN_CHANNELS}",
                        f"mid_channels{MID_CHANNELS}",
                        f"bias{WITH_BIAS}",
                        f"mean{PARAM_MEAN}",
                        f"std{PARAM_STD}",])
    save_path = os.path.join(save_root, exp_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

if __name__ == '__main__':
    seed = 0
    device = 'cuda:0'
    sample_num = 100
    K = KERNEL_SIZE
    H,W = IMAGE_SHAPE

    save_root = '/data2/tangling/conv-generator/outs/corollary2'
    data_path = '/data2/tangling/conv-generator/data/broden1_224/image_14.pt'
    save_path = make_dirs(save_root)
    set_logger(save_path)
    logger = get_logger(__name__,True)
    set_random(seed)
    save_current_src(save_path,'../src')
    save_current_src(save_path,'../scripts')

    # init the conv net
    conv_net = CircuConvNet(
        KERNEL_SIZE,
        IN_CHANNELS,
        MID_CHANNELS,
        CONV_NUM,
        pad_mode='same',
        with_bias=WITH_BIAS,
        with_relu=False,
    ).to(device)
    conv_net.reset_params(PARAM_MEAN,PARAM_STD)

    inputs = get_data(sample_num,data_path).to(device) #1*C*H*W

    error_lis = []
    cos_lis = []
    for sample_id in tqdm(range(sample_num)):
        #sample the input
        error_lis.append([])
        cos_lis.append([])

        input = inputs[sample_id:sample_id+1]
        output = conv_net(input) #1*C*H*W
        #real_fft
        output.retain_grad()
        L = (output**2).sum()
        conv_net.zero_grad()            
        L.backward()

        #path1
        output_grad = output.grad.detach()
        f_input = torch.fft.fft2(input)[0].detach() #C*H*W
        f_output_grad = (torch.fft.fft2(output_grad)[0]/(H*W)).detach() #C*H*W

        for layer_id in range(0,CONV_NUM):

            #true grad
            weight = conv_net.main[layer_id].conv.weight.detach()
            weight_grad = conv_net.main[layer_id].conv.weight.grad.detach()
            f_weight_grad = kernel_fft(weight_grad,IMAGE_SHAPE,device) /(H*W)

            #cal grad
            if layer_id == 0:
                T2,_ = conv_net.get_freq_trans(IMAGE_SHAPE,[1,CONV_NUM],device)
                cal_f_weight_grad_ = torch.zeros((IN_CHANNELS,MID_CHANNELS,H,W),dtype=torch.complex64,device=device)
                cal_f_weight_grad = torch.zeros((IN_CHANNELS,MID_CHANNELS,H,W),dtype=torch.complex64,device=device)
                for u in range(H):
                    for v in range(W):
                        cal_f_weight_grad_[:,:,u,v] = torch.conj(f_input[:,u,v]).unsqueeze(-1) @ (f_output_grad[:,u,v] @ torch.conj(T2[:,:,u,v])).unsqueeze(0)

            elif layer_id == CONV_NUM-1:
                T1,_ = conv_net.get_freq_trans(IMAGE_SHAPE,[0,CONV_NUM-1],device)
                cal_f_weight_grad_ = torch.zeros((MID_CHANNELS,IN_CHANNELS,H,W),dtype=torch.complex64,device=device)
                cal_f_weight_grad = torch.zeros((MID_CHANNELS,IN_CHANNELS,H,W),dtype=torch.complex64,device=device)
                for u in range(H):
                    for v in range(W):
                        cal_f_weight_grad_[:,:,u,v] = torch.conj(T1[:,:,u,v] @ f_input[:,u,v]).unsqueeze(-1) @ f_output_grad[:,u,v].unsqueeze(0)

            else:
                T1,_ = conv_net.get_freq_trans(IMAGE_SHAPE,[0,layer_id],device)
                T2,_ = conv_net.get_freq_trans(IMAGE_SHAPE,[layer_id+1,CONV_NUM],device)
                cal_f_weight_grad_ = torch.zeros((MID_CHANNELS,MID_CHANNELS,H,W),dtype=torch.complex64,device=device)
                cal_f_weight_grad = torch.zeros((MID_CHANNELS,MID_CHANNELS,H,W),dtype=torch.complex64,device=device)
                for u in range(H):
                    for v in range(W):
                        cal_f_weight_grad_[:,:,u,v] = torch.conj(T1[:,:,u,v] @ f_input[:,u,v]).unsqueeze(-1) @ (f_output_grad[:,u,v] @ torch.conj(T2[:,:,u,v])).unsqueeze(0)
            
            #add the coef chi
            for u in range(H):
                for v in range(W):
                    delta_uv = delta_trans(KERNEL_SIZE,IMAGE_SHAPE,[u,v],device)
                    cal_f_weight_grad += cal_f_weight_grad_[:,:,u,v].unsqueeze(-1).unsqueeze(-1) * delta_uv.unsqueeze(0).unsqueeze(0)

            cal_f_weight_grad =  torch.transpose(cal_f_weight_grad,0,1)
            error = get_error(f_weight_grad.cpu().numpy(),cal_f_weight_grad.cpu().numpy())
            cos = get_cos(f_weight_grad.cpu().numpy(),cal_f_weight_grad.cpu().numpy(),dims=(-2,-1))
            error_lis[sample_id].append(error)
            cos_lis[sample_id].append(cos)


    mean_error = np.array(error_lis).mean(0)
    std_error = np.array(error_lis).std(0)
    np.save(os.path.join(save_path,'mean_error.npy'),mean_error)
    np.save(os.path.join(save_path,'std_error.npy'),std_error)
    logger.info(f'mean error:{mean_error}')
    logger.info(f'std error:{std_error}')

    mean_cos = np.array(cos_lis).mean(0)
    std_cos = np.array(cos_lis).std(0)
    np.save(os.path.join(save_path,'mean_cos.npy'),mean_cos)
    np.save(os.path.join(save_path,'std_cos.npy'),std_cos)
    logger.info(f'mean cos:{mean_cos}')
    logger.info(f'std cos:{std_cos}')


    #     #forward
    #     for i in range(layer_id):
    #         weights = conv_net.main[i].conv.weight.detach()
    #         bias = conv_net.main[i].conv.bias.detach()
    #         bias_fft = bias * IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
    #         weight_fft = kernel_fft(weights,IMAGE_SHAPE)

    #         for j in range(IMAGE_SHAPE[0]):
    #             for k in range(IMAGE_SHAPE[1]):
    #                 if j == 0 and k == 0:
    #                     forward_fft[:,j,k] = torch.conj(weight_fft[:,:,j,k]) @ forward_fft[:,j,k] + bias_fft
    #                 else:
    #                     forward_fft[:,j,k] = torch.conj(weight_fft[:,:,j,k]) @ forward_fft[:,j,k]

    #     #backward
    #     for i in range(CONV_NUM-1,layer_id,-1):
    #         weights = conv_net.main[i].conv.weight.detach()
    #         weight_fft = kernel_fft(weights,IMAGE_SHAPE)
    #         for j in range(IMAGE_SHAPE[0]):
    #             for k in range(IMAGE_SHAPE[1]):
    #                 backward_fft[:,:,j,k] = backward_fft[:,:,j,k] @ torch.conj(weight_fft[:,:,j,k])

    #     #real grad
    #     # real_grad = kernel_fft(weights_grad,IMAGE_SHAPE) / (IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    #     # real_grad =  torch.transpose(real_grad,0,1)

    #     #calucate grad
    #     grad_uncons = torch.zeros(MID_DIM,MID_DIM,*IMAGE_SHAPE,dtype=torch.complex64)
    #     grad_cons = torch.zeros(MID_DIM,MID_DIM,*IMAGE_SHAPE,dtype=torch.complex64)
    #     for j in range(IMAGE_SHAPE[0]):
    #         for k in range(IMAGE_SHAPE[1]):
    #             grad_uncons[:,:,j,k] = forward_fft[:,j,k].unsqueeze(1) @ backward_fft[:,:,j,k]

    #     for u in range(IMAGE_SHAPE[0]):
    #         for v in range(IMAGE_SHAPE[1]):
    #             beta_uv = torch.torch.from_numpy(delta_trans(KERNEL_SIZE,IMAGE_SHAPE,[u,v]))
    #             grad_cons += grad_uncons[:,:,u,v].unsqueeze(-1).unsqueeze(-1) * beta_uv.unsqueeze(0).unsqueeze(0)

    #     #verify
    #     weight_fft = kernel_fft(weights,IMAGE_SHAPE)
    #     weight_fft += torch.transpose(grad_cons,0,1)
    #     iweights = kernel_ifft(weight_fft,KERNEL_SIZE)
    #     delta_weight = (iweights - weights)*(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
    #     delta_weight_real = weights_grad

    #     delta = (delta_weight - delta_weight_real)
    #     delta_norm = np.linalg.norm(delta.numpy().reshape(-1),ord=2)
    #     grad_norm = np.linalg.norm(delta_weight_real.numpy().reshape(-1),ord=2)

    # print(delta_norm/grad_norm)

    #compare
    # delta = (real_grad - grad_cons)
    # delta_norm = np.linalg.norm(delta.numpy().reshape(-1),ord=2)
    # grad_norm = np.linalg.norm(real_grad.numpy().reshape(-1),ord=2)

    # print(delta_norm/grad_norm)