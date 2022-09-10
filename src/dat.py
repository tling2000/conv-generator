import torch

def get_gaussian_2d(
    mean: float,
    std: float,
    size: list,
    )->torch.Tensor:
    assert len(size) == 4,'must be 4-d tensor'
    dat = torch.randn(size) * std + mean
    return dat.detach()