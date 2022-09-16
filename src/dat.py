import torch

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