from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import random
import logging

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
                 vmax: int = None) -> None: 
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
    fig, ax = plt.subplots(figsize=(10, 8))
    if col == "coolwarm": 
        ax = sns.heatmap(mat, annot=False, cbar=True, cmap = col, vmin = vmin, vmax = vmax) 
    else: 
        ax = sns.heatmap(mat, annot=False, cbar=True, vmin = vmin, vmax = vmax) 
    ax.set_title(name) 
    plt.axis('off')
    fig.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if vmin is None:
        path = os.path.join(save_path, f"{name}.png")
    else:
        path = os.path.join(save_path, "{}_{:.4f}to{:.4f}.png".format(name, vmin, vmax)) 
    fig.savefig(path) 
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

def get_delta(true_value: np.ndarray,
              cal_value: np.ndarray,):
    delta = true_value - cal_value
    delta_norm = np.linalg.norm(delta.reshape(-1),ord=2)
    true_norm = np.linalg.norm(true_value.reshape(-1),ord=2)
    return delta_norm / true_norm