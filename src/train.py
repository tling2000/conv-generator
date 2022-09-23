import os
import torch
import torch.nn as nn
import numpy as np
from utils import get_logger, plot_heatmap,save_image,plot_sub_heatmap,get_fft,get_tensor
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from hook import Hook

def train(
    model: nn.Module,
    Xs: torch.Tensor,
    save_path: str,
    rounds: int,
    batch_size: int,
    lr: float,
    device: str,
    trace_ids: list,
    ):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path)
    
    #basic settings
    N,C,H,W = Xs.size()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    #hook on the model
    layer_names = [
        'decode/0',
        'decode/2',
        'decode/4',
    ]
    hook = Hook(model,layer_names)

    # outputs_dict = {}
    # inputs_dict = {}
    # for trace_id in trace_ids:
    #     inputs_dict[f'sample{trace_id}'] = Xs[trace_id].detach()
    #     outputs_dict[f'sample{trace_id}'] = {}
    #     for layer_name in layer_names:
    #         outputs_dict[f'sample{trace_id}'][layer_name] = []

    for trace_id in trace_ids:
        image = Xs[trace_id].detach()
        f_out = get_fft(image,no_basis=False,is_cut=False)
        f_out_nozero = get_fft(image,no_basis=True,is_cut=False)
        save_image(os.path.join(save_path,'results'),image,f'sample{trace_id}_image',is_rgb=True)
        save_image(os.path.join(save_path,'results_nozero'),image,f'sample{trace_id}_image',is_rgb=True)
        plot_sub_heatmap(os.path.join(save_path,'results'),[f_out],f'sample{trace_id}_spectrum')
        plot_sub_heatmap(os.path.join(save_path,'results_nozero'),[f_out_nozero],f'sample{trace_id}_spectrum')

    #train
    losses = []
    model.train()
    for rnd in range(rounds):
        indexs = np.arange(N)
        np.random.shuffle(indexs)
        batch_num = int(round(N/batch_size))
        batches = np.array_split(indexs,batch_num)
        loss = 0

        for batch_id,batch in enumerate(tqdm(batches)):
            X_bs = Xs[batch].detach().to(device)
            X_bs_hat  =  X_bs.detach()
            optimizer.zero_grad()
            X_bs_pre = model(X_bs)
            bloss = criterion(X_bs_pre,X_bs_hat)
            bloss.backward()
            optimizer.step()

            logger.info(f'batch loss: {bloss.item()}')
            loss += bloss.item()/batch_num

            #trace the samples
            insert_pixcel = 1

            features_dict = hook.get_features()
            for trace_id  in trace_ids:
                if trace_id in batch:

                    outs = []
                    f_outs = []
                    f_outs_nozero = []

                    trace_batch_id  = np.argwhere(batch==trace_id)[0][0]
                    for layer_name in features_dict.keys():

                        image = torch.from_numpy(features_dict[layer_name][trace_batch_id])
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
                    save_image(os.path.join(save_path,'results'),out,f'epoch{rnd}-sample{trace_id}-uptime',is_rgb=True)
                    save_image(os.path.join(save_path,'results_nozero'),out,f'epoch{rnd}-sample{trace_id}-uptime',is_rgb=True)

                    plot_sub_heatmap(os.path.join(save_path,'results'),f_outs,f'epoch{rnd}-sample{trace_id}-upspec',cbar=False)
                    plot_sub_heatmap(os.path.join(save_path,'results_nozero'),f'epoch{rnd}-sample{trace_id}-upspec',cbar=False)

        #register the loss
        losses.append(loss)

        #plot the heatmap
        if rnd == rounds-1:
            model_path = os.path.join(save_path,'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(),os.path.join(model_path,f'model_{rnd}.pt'))

    fig,ax = plt.subplots()
    ax.plot(range(1,rounds+1),losses)
    fig.savefig(os.path.join(save_path,'loss.jpg'))
    np.save(os.path.join(save_path,'losses.npy'),np.array(losses))
