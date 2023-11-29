from matplotlib import pyplot as plt
import numpy as np
import os

def plot_mean_std(ax,index,mean,std,color,label,ls):
    index = np.array(index) + 1
    r1 = list(map(lambda x: x[0]-x[1], zip(mean, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(mean, std)))#下方差
    ax.plot(index, mean, color=color,label=label,linewidth=2.0,ls=ls)
    ax.fill_between(index, r1, r2, color=color, alpha=0.2)
    return True

def plot_figure(root1,root2,label1,label2,save_root,name,index):
    figsize = (3,2.2)
    
    fig,ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Network depth L',fontdict={'size':16})
    ax.set_ylabel('Cosine similarity',fontdict={'size':16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(True, which = 'both',linestyle='--')

    mean_c = np.load(os.path.join(root1,'mean_cos.npy'))[index]
    std_c = np.load(os.path.join(root1,'std_cos.npy'))[index]
    plot_mean_std(ax,index,mean_c,std_c,'C0',label1,ls='--')

    mean_c = np.load(os.path.join(root2,'mean_cos.npy'))[index]
    std_c = np.load(os.path.join(root2,'std_cos.npy'))[index]
    plot_mean_std(ax,index,mean_c,std_c,'C1',label2,ls='-')

    ax.legend(loc=4)
    fig.savefig(os.path.join(save_root,f'{name}.jpg'),dpi=300,bbox_inches='tight')
    return True


if __name__ == '__main__':
    label1 = 'w/o ReLU'
    label2 = 'with ReLU'
    tag = 'cifar'

    # corollary1
    # root1 = f'/data2/tangling/conv-generator/outs/corollary1/0921/{tag}-worelu'
    # root2 = f'/data2/tangling/conv-generator/outs/corollary1/0921/{tag}-exp'
    # save_root = '/data2/tangling/conv-generator/outs/corollary1/0921'
    # name = f'{tag}'
    # plot_figure(root1,root2,label1,label2,save_root,name,range(9))

    # corollary2
    root1 = f'/data2/tangling/conv-generator/outs/corollary2/0922/{tag}-worelu'
    root2 = f'/data2/tangling/conv-generator/outs/corollary2/0922/{tag}-relu'
    save_root = f'/data2/tangling/conv-generator/outs/corollary2/0922'
    name = f'{tag}'
    plot_figure(root1,root2,label1,label2,save_root,name,range(1,10))

