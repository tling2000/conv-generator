from matplotlib import pyplot as plt
import numpy as np
import os

def plot_mean_std(ax,mean,std,color,label,ls):
    iters = range(1,len(mean)+1)
    r1 = list(map(lambda x: x[0]-x[1], zip(mean, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(mean, std)))#下方差
    ax.plot(iters, mean, color=color,label=label,linewidth=2.0,ls=ls)
    ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    return True


if __name__ == '__main__':
    figsize = (3.2,2.2)
    labels = [
        'without_zeropadding',
        'with_reluzeropadding'
    ]
    colors = [
        'C0',
        'C1'
    ]

    tag = 'corollary1'
    save_root = '/data2/tangling/conv-generator/outs/corollary1'
    dat_roots = [
        '/data2/tangling/conv-generator/outs/corollary1/0919_2/0919-125304-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
        '/data2/tangling/conv-generator/outs/corollary1/0919-151148-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    ]
    #draw cos
    fig,ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Network depth L')
    ax.set_ylabel('Cosine similarity')
    ax.grid(True, which = 'both',linestyle='--')
    for i in range(len(dat_roots)):
        dat_root = dat_roots[i]
        mean_c = np.load(os.path.join(dat_root,'mean_cos.npy'),)
        std_c = np.load(os.path.join(dat_root,'std_cos.npy'))
        if i == 0:
            ls = '--'
        else:
            ls = '-'
        plot_mean_std(ax,mean_c,std_c,colors[i],labels[i],ls=ls)
        
    ax.legend()
    fig.savefig(os.path.join(save_root,'cos.jpg'),dpi=300,bbox_inches='tight')

    # tag = 'corollary2'
    # save_root = '/data2/tangling/conv-generator/outs/corollary2'
    # dat_roots = [
    #     '/data2/tangling/conv-generator/outs/corollary2/0919-082559-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
    #     '/data2/tangling/conv-generator/outs/corollary2/0919-083439-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
    # ]
    # #draw cos
    # fig,ax = plt.subplots(figsize=figsize)
    # ax.set_xlabel('Network depth L')
    # ax.set_ylabel('Cosine similarity')
    # ax.grid(True, which = 'both',linestyle='--')
    # for i in range(len(dat_roots)):
    #     dat_root = dat_roots[i]
    #     mean_c = np.load(os.path.join(dat_root,'mean_cos.npy'))
    #     std_c = np.load(os.path.join(dat_root,'std_cos.npy'))
    #     if i == 0:
    #         ls = '--'
    #     else:
    #         ls = '-'
    #     plot_mean_std(ax,mean_c,std_c,colors[i],labels[i],ls='--')
    # ax.legend()
    # fig.savefig(os.path.join(save_root,'cos.jpg'),dpi=300,bbox_inches='tight')

