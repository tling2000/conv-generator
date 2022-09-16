from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import random
import logging
from torchvision import transforms

def set_logger(save_path: str) -> None:
    logfile = os.path.join(save_path, "logfile.txt")
    logging.basicConfig(filename=logfile,
                        filemode="w+",
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(
        '%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger().addHandler(console)


def get_logger(name:str,
               verbose:bool = True) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    if not verbose:
        logger.setLevel(logging.INFO)
    return logger

def set_random(seed):
    """
    set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_heatmap(save_path: str, 
                 mat: np.ndarray, 
                 name: str, 
                 col: str = 'coolwarm', 
                 vmin: int = None, 
                 vmax: int = None,
                 cbar: bool = True,
                 ) -> None: 
    """plot heatmap 
    Args: 
        save_path (str): save_path
        mat: the mat used to plot heatmap numpy 2darray 
        name (str): the name of the plot. 
        col (str): name of color 
        vmin (int): the min value of the colorbar. default is None.
        vmax (int): the max value of the colorbar. default is None.
    
    Returns:
        None
    """
    assert (vmin is None) == (vmax is None), "vmin and vmax must be both None or not None"
    if cbar:
        fig, ax = plt.subplots(figsize=(3,2.4))
    else:
        fig, ax = plt.subplots(figsize=(3,3))

    if col == "coolwarm": 
        ax = sns.heatmap(mat, annot=False, cbar=cbar, cmap = col, vmin = vmin, vmax = vmax) 
    else: 
        ax = sns.heatmap(mat, annot=False, cbar=cbar, vmin = vmin, vmax = vmax) 
    # ax.set_title(name) 
    plt.axis('off')
    fig.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if vmin is None:
        path = os.path.join(save_path, f"{name}.png")
    else:
        path = os.path.join(save_path, "{}_{:.4f}to{:.4f}.png".format(name, vmin, vmax)) 
    fig.savefig(path,dpi=300) 
    plt.close()

def save_current_src(save_path: str,
                     src_path: str) -> None:
    """save the current src.
    Args:
        save_path (str): the path to save the current src.
        src_path (str): the path to the current src.
    Returns:
        None
    """
    logger = get_logger(__name__)
    logger.info("save the current src")
    os.system("cp -r {} {}".format(src_path, save_path))

def get_error(true_value: np.ndarray,
              cal_value: np.ndarray,):
    delta = true_value - cal_value
    delta_norm = np.linalg.norm(delta.reshape(-1),ord=2)
    true_norm = np.linalg.norm(true_value.reshape(-1),ord=2)
    return delta_norm / true_norm

def get_mean_freq_scale(image):
    assert len(image.shape) == 3
    H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    f_image = torch.fft.fftshift(f_image,dim = (-2,-1))
    f_image_power = torch.real(f_image * torch.conj(f_image))
    f_image_norm = f_image_power / torch.sum(f_image_power,dim=(-2,-1),keepdim=True)
    low_freq_scale = torch.sum(f_image_norm[:,int(3*H/8):int(5*H/8)],dim=(-2,-1))
    mean_scale = low_freq_scale.mean()
    return mean_scale

def plot_fft(save_path,image,name):
    assert len(image.shape) == 2,''
    H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    # f_image[0,0] = 0
    f_image = torch.fft.fftshift(f_image)
    f_image_power = torch.real(f_image * torch.conj(f_image))
    f_image_norm = f_image_power / f_image_power.sum()
    plot_heatmap(save_path,torch.log10(f_image_norm),name,vmin=-6,vmax=0,cbar=False)
    low_freq_scale = torch.sum(f_image_norm[int(3*H/8):int(5*H/8)])
    return low_freq_scale

def save_image(save_path,tensor,name):
    assert len(tensor.shape) == 2,''
    tensor = tensor.detach().cpu()
    tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    unloader = transforms.ToPILImage()
    image = unloader(tensor)
    image.save(os.path.join(save_path,f'{name}.jpg'))