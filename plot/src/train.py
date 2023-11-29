import os
import torch
import torch.nn as nn
import numpy as np
from utils import get_logger, plot_heatmap,plot_fft,save_image
import pandas as pd
from config import IMAGE_SHAPE,KERNEL_SIZE
from tqdm import tqdm
from matplotlib import pyplot as plt

def train(
    model: nn.Module,
    Xs: torch.Tensor,
    save_path: str,
    rounds: int,
    batch_size: int,
    lr: float,
    device: str,
    ):

    N,C,H,W = Xs.size()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger = get_logger(save_path)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.MSELoss()

    losses = []
    trace_idx =  [0,1,2,3,4]

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
            for trace_id  in trace_idx:
                if trace_id in batch:
                    trace_batch_id  = np.argwhere(batch==trace_id)[0][0]

                    if rnd == 0:
                        X_hat = X_bs[trace_batch_id].mean(0).detach()
                        save_image(os.path.join(save_path,f'trace_{trace_id}'),X_hat,f'X_hat',False)
                        plot_fft(os.path.join(save_path,f'trace_{trace_id}'),X_hat,f'f_X_hat')
                    if rnd % 1 == 0:
                        X_pre = X_bs_pre[trace_batch_id].mean(0).detach()
                        save_image(os.path.join(save_path,f'trace_{trace_id}'),X_pre,f'X_pre_rnd{rnd}',False)
                        plot_fft(os.path.join(save_path,f'trace_{trace_id}'),X_pre,f'f_X_pre{rnd}')
        #register the loss
        losses.append(loss)

        #plot the heatmap
        if rnd % 20 == 0 or rnd == rounds-1:
            model_path = os.path.join(save_path,'model')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(),os.path.join(model_path,f'model_{rnd}.pt'))

    fig,ax = plt.subplots()
    ax.plot(range(1,rounds+1),losses)
    fig.savefig(os.path.join(save_path,'loss.jpg'))
    np.save(os.path.join(save_path,'losses.npy'),np.array(losses))

