import numpy as np
from matplotlib import pyplot as plt
import os
from utils import plot_heatmap
from tqdm import tqdm

if __name__ == '__main__':
    M = 200
    N = 200
    # u = 40
    # v = 76
    K = 3

    lis  = []
    for u in range(M-K+1):
        for v in range(M-K+1):
            x = np.arange(M-K+1)
            y = np.arange(N-K+1)
            X,Y = np.meshgrid(x,y)

            if u == 0 or u == M //2 :
                heatx = np.ones_like(X)
            else: 
                heatx = np.abs(np.sin(u*(K-1)*np.pi/M)/np.sin((u*(-M+K-1)+X*M)*np.pi/M/(M-K+1))) /M

            if v == 0 or v == N//2:
                heaty = np.ones_like(Y)
            else:
                heaty = np.abs(np.sin(v*(K-1)*np.pi/N)/np.sin((v*(-N+K-1)+Y*N)*np.pi/N/(N-K+1))) /N

            heat = heatx*heaty
            # print(u,v)
            lis.append(heat[v,u])
            
            # save_path = '/data2/tangling/conv-expression/outs/lab7/alpha'

            # plot_heatmap(save_path,heat,f'u{u}_v{v}.jpg',vmin=0,vmax=1)
    array = np.array(lis)
    print(array.mean())
    print(array.std())
        # fig,ax = plt.subplots()
        # ax.plot(x,y)
        # ax.grid()
        # ax.set_xlabel(f'u\'|u={u}')
        # ax.set_ylabel('|alpha|')
        # ax.set_xlim([0,M-K])
        # fig.savefig(os.path.join(save_path,f'u{u}.jpg'))
