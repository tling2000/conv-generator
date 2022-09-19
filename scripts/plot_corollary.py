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

def plot_figure(root1,root2,label1,label2,save_root):
    figsize = (3,2)
    
    fig,ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Network depth L')
    ax.set_ylabel('Cosine similarity')
    ax.grid(True, which = 'both',linestyle='--')

    mean_c = np.load(os.path.join(root1,'mean_cos.npy'))[:-1]
    std_c = np.load(os.path.join(root1,'std_cos.npy'))[:-1]
    plot_mean_std(ax,mean_c,std_c,'C0',label1,ls='--')

    mean_c = np.load(os.path.join(root2,'mean_cos.npy'))[:-1]
    std_c = np.load(os.path.join(root2,'std_cos.npy'))[:-1]
    plot_mean_std(ax,mean_c,std_c,'C1',label2,ls='-')

    ax.legend()
    fig.savefig(os.path.join(save_root,'cos.jpg'),dpi=300,bbox_inches='tight')
    return True


if __name__ == '__main__':
    label1 = 'without_relu_zero'
    label2 = 'with_relu_zero'

    # corollary1
    root1 = '/data2/tangling/conv-generator/outs/corollary1/0919_2/0919-125304-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    root2 = '/data2/tangling/conv-generator/outs/corollary1/0919_2/0919-125330-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    save_root = '/data2/tangling/conv-generator/outs/corollary1'
    plot_figure(root1,root2,label1,label2,save_root)

    # corollary2
    root1 = '/data2/tangling/conv-generator/outs/corollary2/0919/0919-082559-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    root2 = '/data2/tangling/conv-generator/outs/corollary2/0919/0919-082746-conv_num10-K3-in_channels3-mid_channels16-biasFalse-mean0-std0.1'
    save_root = '/data2/tangling/conv-generator/outs/corollary2'
    plot_figure(root1,root2,label1,label2,save_root)

