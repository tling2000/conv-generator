
import os,sys
from turtle import forward
sys.path.append('../src')

from coef import delta_trans
import torch

H,W = 224,224
K = 3
delta_uv = torch.zeros((H,W,H,W),dtype=torch.complex64)
for u in range(H):
    print(u)
    for v in range(W):
        delta_uv[u,v] = delta_trans(K,(H,W),(u,v),'cpu')
torch.save(delta_uv,f'/data2/tangling/conv-generator/data/delta_trans/chi_H{H}_K{K}.pt')