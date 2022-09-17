from xml.etree.ElementTree import PI
import torch
from torchvision import transforms
import os
from PIL import Image

def get_gaussian_2d(
    mean: float,
    std: float,
    size: list,
    )->torch.Tensor:
    assert len(size) == 4,'must be 4-d tensor'
    dat = torch.randn(size) * std + mean
    return dat.detach()

def get_tiny_imagenet(
    sample_num: int,
    ):
    dat = torch.load('/data2/tangling/conv-generator/data/tiny-imagenet/image.pt')
    dat = dat[:sample_num]
    return dat.detach()

def create_data(
    img_shape: list,
    data_num: int,
    data_path:str,
    save_path: str,
    ):
    data_transform = transforms.Compose([
        transforms.Resize(img_shape),
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
    torch.save(Xs,os.path.join(save_path,'image.pt'))
    return True