import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image

def get_gaussian_2d(
    mean: float,
    std: float,
    size: list,
    )->torch.Tensor:
    assert len(size) == 4,'must be 4-d tensor'
    dat = torch.randn(size) * std + mean
    return dat.detach()

def get_data(
    sample_num: int,
    data_path: str,
    ):
    dat = torch.load(data_path)
    if sample_num == None:
        return dat.detach()
    dat = dat[:sample_num]
    return dat.detach()

def create_data(
    img_shape: list,
    data_num: int,
    data_path:str,
    save_path: str,
    ):
    data_transform = transforms.Compose([
        transforms.CenterCrop(img_shape[0]),
        # transforms.Resize(img_shape),
        transforms.ToTensor(),
    ])
    
    data_list = []
    for root,dirs,files in os.walk(data_path):
        for name in files:
            if name.endswith('jpg') or name.endswith('png') or name.endswith('JPEG'):
                if 'color' in name:
                    continue
                image_path = os.path.join(root,name)

                image = Image.open(image_path).convert('RGB')
                image = data_transform(image)
                data_list.append(image)

                if len(data_list)%100==0:
                    print(len(data_list))
                if len(data_list) == data_num:
                    break

    Xs = torch.stack(data_list).detach()
    torch.save(Xs,os.path.join(save_path,f'image_{img_shape[0]}.pt'))
    return True

def create_cifar_data(
    data_path,
    save_path,
    ):
    import pickle
    with open(data_path,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    Xs = dict[b'data']
    Xs = Xs.reshape(len(Xs),3,32,32)
    Xs = Xs.astype(np.float32)/255
    Xs = torch.from_numpy(Xs)
    torch.save(Xs,os.path.join(save_path,f'image.pt'))
    return True



if __name__ == '__main__':
    create_cifar_data(
        '/data2/tangling/conv-generator/data/cifar-10-batches-py/data_batch_1',
        '/data2/tangling/conv-generator/data/cifar-10-batches-py'
    )
