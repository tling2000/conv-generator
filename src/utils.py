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
    # if cbar:
    #     fig, ax = plt.subplots(figsize=(3,2.4))
    # else:
    #     fig, ax = plt.subplots(figsize=(3,3))
    fig, ax = plt.subplots()
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

def plot_sub_heatmap(save_path: str, 
                 mats: list, 
                 name: str, 
                 vmin: int = None, 
                 vmax: int = None,
                 cbar: bool = True,
                 ) -> None: 
    assert (vmin is None) == (vmax is None), "vmin and vmax must be both None or not None"

    feature_num = len(mats)
    fig, ax = plt.subplots(1,feature_num,figsize=(feature_num*4,4))
    if feature_num == 1:
        sns.heatmap(mats[0], annot=False, cbar=cbar, cmap = 'coolwarm', vmin = vmin, vmax = vmax,ax=ax) 
        ax.set_axis_off()  
    else:
        for i in range(feature_num):
            sns.heatmap(mats[i], annot=False, cbar=cbar, cmap = 'coolwarm', vmin = vmin, vmax = vmax,ax=ax[i]) 
            ax[i].set_axis_off()  
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

def get_error(
        true_value: np.ndarray,
        cal_value: np.ndarray,
        ):
    true_value,cal_value = np.abs(true_value),np.abs(cal_value)
    delta = true_value - cal_value
    delta_norm = np.linalg.norm(delta.reshape(-1),ord=2)
    true_norm = np.linalg.norm(true_value.reshape(-1),ord=2)
    error = delta_norm / true_norm

    return error

def get_cos(
        true_value: np.ndarray,
        cal_value: np.ndarray,
        dims: list,
        ):
    true_value,cal_value = np.abs(true_value),np.abs(cal_value)
    num = np.sum((true_value * cal_value),axis=dims)
    denom = np.sqrt(np.sum(true_value**2,axis=dims)) * np.sqrt(np.sum(cal_value**2,axis=dims))
    cos = num / denom
    cos [denom == 0] = 1
    # cos = cos[denom!=0]
    return cos.mean()

def get_fft(image,no_basis,is_cut,cut_scale=None):
    assert len(image.shape) == 3,''
    C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    if no_basis:
        f_image[:,0,0] = 0
    f_image = torch.fft.fftshift(f_image,dim=(-2,-1))
    f_image_norm = torch.abs(f_image).mean(0)
    if no_basis:
        f_image_norm[H//2,W//2] = f_image_norm.max()
    
    if is_cut:
        assert cut_scale != None
        f_image_norm = f_image_norm[int(H/2-H/(2*cut_scale)):int(H/2+H/(2*cut_scale)),int(W/2-W/(2*cut_scale)):int(W/2+H/(2*cut_scale))]
    f_image_norm = (f_image_norm-f_image_norm.min()) /(f_image_norm.max()-f_image_norm.min())
    return f_image_norm

def get_tensor(tensor,):
    tensor = tensor.mean(0)
    tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    return tensor

def save_image(save_path,tensor,name,is_norm=False,is_rgb=False):
    assert len(tensor.shape) == 3,''
    tensor = tensor.detach().cpu()
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_rgb:
        pass
    else:
        tensor = tensor.mean(0)
    if is_norm:
        tensor = (tensor-tensor.min()) /(tensor.max()-tensor.min())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    unloader = transforms.ToPILImage()
    image = unloader(tensor)
    
    image.save(os.path.join(save_path,f'{name}.jpg'),quality=100)


def get_low_freq_scale(image):
    assert len(image.shape) == 3
    C,H,W = image.shape
    image = image.detach().cpu()
    f_image = torch.fft.fft2(image)
    f_image = torch.fft.fftshift(f_image,dim = (-2,-1))
    f_image_power = torch.real(f_image * torch.conj(f_image))
    f_image_norm = f_image_power / torch.sum(f_image_power,dim=(-2,-1),keepdim=True)
    low_freq_scale = torch.sum(f_image_norm[:,int(H/2-H/8):int(H/2+H/8),int(W/2-W/8):int(W/2+H/8)],dim=(-2,-1))
    return low_freq_scale