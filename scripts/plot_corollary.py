from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

def plot_mean_std(ax,index,mean,std,color,label,ls):
    index = np.array(index) + 1
    r1 = list(map(lambda x: x[0]-x[1], zip(mean, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(mean, std)))#下方差
    ax.plot(index, mean, color=color,label=label,linewidth=2.0,ls=ls)
    ax.fill_between(index, r1, r2, color=color, alpha=0.2)
    return True

def plot_figure(root1,root2,root3,label1,label2,label3,save_root,name,index):
    figsize = (3,2.6)
    # y_major_locator=MultipleLocator(0.05)
    
    fig,ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Depth of the network',fontdict={'size':16})
    ax.set_ylabel('Cosine similarity',fontdict={'size':16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.grid(True, which = 'both',linestyle='--')
    # ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    mean_c = np.load(os.path.join(root3,'mean_cos.npy'))[index]
    std_c = np.load(os.path.join(root3,'std_cos.npy'))[index]
    plot_mean_std(ax,index,mean_c,std_c,'C2',label3,ls='-')

    mean_c = np.load(os.path.join(root2,'mean_cos.npy'))[index]
    std_c = np.load(os.path.join(root2,'std_cos.npy'))[index]
    plot_mean_std(ax,index,mean_c,std_c,'C1',label2,ls='--')

    mean_c = np.load(os.path.join(root1,'mean_cos.npy'))[index]
    std_c = np.load(os.path.join(root1,'std_cos.npy'))[index]
    plot_mean_std(ax,index,mean_c,std_c,'C0',label1,ls='-.')

    num1 = 1.02
    num2 = 0
    num3 = 3
    num4 = 0
    ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4,fontsize = 16)
    fig.savefig(os.path.join(save_root,f'{name}.pdf'),bbox_inches='tight')
    return True


if __name__ == '__main__':
    label1 = 'baseline3\nw/o zero padding, w/o ReLU'
    label2 = 'baseline2\nw/ zero padding, w/o ReLU'
    label3 = 'baseline1\nw/ zero padding, w/ ReLU'
    tag = 'broden'

    # corollary1
    root1 = f'/data2/tangling/conv-generator/outs/corollary1/0922/{tag}-theo'
    root2 = f'/data2/tangling/conv-generator/outs/corollary1/0922/{tag}-worelu'
    root3 = f'/data2/tangling/conv-generator/outs/corollary1/0922/{tag}-relu'
    save_root = '/data2/tangling/conv-generator/outs/corollary1/0118'
    name = f'{tag}'
    plot_figure(root1,root2,root3,label1,label2,label3,save_root,name,range(9))

    # corollary2
    root1 = f'/data2/tangling/conv-generator/outs/corollary2/0924/{tag}-theo'
    root2 = f'/data2/tangling/conv-generator/outs/corollary2/0924/{tag}-worelu'
    root3 = f'/data2/tangling/conv-generator/outs/corollary2/0924/{tag}-relu'
    save_root = f'/data2/tangling/conv-generator/outs/corollary2/0118'
    name = f'{tag}'
    plot_figure(root1,root2,root3,label1,label2,label3,save_root,name,range(1,10))