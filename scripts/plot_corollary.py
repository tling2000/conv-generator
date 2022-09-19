from matplotlib import pyplot as plt
import numpy as np
import os

def plot_mean_std(ax,mean,std,color,label):
    iters = range(1,len(mean)+1)
    r1 = list(map(lambda x: x[0]-x[1], zip(mean, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(mean, std)))#下方差
    ax.plot(iters, mean, color=color,label=label,linewidth=3.0)
    ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    return True


if __name__ == '__main__':
    # tag = 'corollary1'
    # save_root = '/data2/tangling/conv-generator/outs/corollary1'
    # dat_roots = [
    #     '/data2/tangling/conv-generator/outs/corollary1/0919/0919-002026-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
    #     '/data2/tangling/conv-generator/outs/corollary1/0919/0919-002153-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    # ]
    # labels = [
    #     'with_relu',
    #     'without_relu'
    # ]
    # colors = [
    #     'C1',
    #     'C0'
    # ]

    # #draw error
    # fig,ax = plt.subplots(figsize=(9,6))
    # ax.set_xlabel('Network depth L')
    # ax.set_ylabel('Error')
    # ax.grid(True, which = 'both',linestyle='--')
    # for i in range(len(dat_roots)):
    #     dat_root = dat_roots[i]
    #     mean_e = np.load(os.path.join(dat_root,'mean_error.npy'))
    #     std_e = np.load(os.path.join(dat_root,'std_error.npy'))
    #     plot_mean_std(ax,mean_e,std_e,colors[i],labels[i])
    # ax.legend()
    # fig.savefig(os.path.join(save_root,'error.jpg'),dpi=300,bbox_inches='tight')

        
    # #draw cos
    # fig,ax = plt.subplots(figsize=(9,6))
    # ax.set_xlabel('Network depth L')
    # ax.set_ylabel('Cos')
    # ax.grid(True, which = 'both',linestyle='--')
    # for i in range(len(dat_roots)):
    #     dat_root = dat_roots[i]
    #     mean_c = np.load(os.path.join(dat_root,'mean_cos.npy'))
    #     std_c = np.load(os.path.join(dat_root,'std_cos.npy'))
    #     plot_mean_std(ax,mean_c,std_c,colors[i],labels[i])
    # ax.legend()
    # fig.savefig(os.path.join(save_root,'cos.jpg'),dpi=300,bbox_inches='tight')

    tag = 'corollary2'
    save_root = '/data2/tangling/conv-generator/outs/corollary2'
    dat_roots = [
        '/data2/tangling/conv-generator/outs/corollary2/0919-083439-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
        '/data2/tangling/conv-generator/outs/corollary2/0919-082559-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
        '/data2/tangling/conv-generator/outs/corollary2/0919-082746-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
        '/data2/tangling/conv-generator/outs/corollary2/0919-083225-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1',
    ]
    labels = [
        'without_relu',
        'with_relu_cos0',
        'with_relu_cos1',
        'with_relu_cos_ignore',
    ]
    colors = [
        'C0',
        'C1',
        'C2',
        'C3',
    ]

    #draw error
    fig,ax = plt.subplots(figsize=(9,6))
    ax.set_xlabel('Network depth L')
    ax.set_ylabel('Error')
    ax.grid(True, which = 'both',linestyle='--')
    for i in range(len(dat_roots)):
        dat_root = dat_roots[i]
        mean_e = np.load(os.path.join(dat_root,'mean_error.npy'))
        std_e = np.load(os.path.join(dat_root,'std_error.npy'))
        plot_mean_std(ax,mean_e,std_e,colors[i],labels[i])
    ax.legend()
    fig.savefig(os.path.join(save_root,'error.jpg'),dpi=300,bbox_inches='tight')

        
    #draw cos
    fig,ax = plt.subplots(figsize=(9,6))
    ax.set_xlabel('Network depth L')
    ax.set_ylabel('Cos')
    ax.grid(True, which = 'both',linestyle='--')
    for i in range(len(dat_roots)):
        dat_root = dat_roots[i]
        mean_c = np.load(os.path.join(dat_root,'mean_cos.npy'))
        std_c = np.load(os.path.join(dat_root,'std_cos.npy'))
        plot_mean_std(ax,mean_c,std_c,colors[i],labels[i])
    ax.legend()
    fig.savefig(os.path.join(save_root,'cos.jpg'),dpi=300,bbox_inches='tight')

