# Conv Generator

## Abstract

Investigate the expression bottleneck of convolutional generative networks.


## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n Conv python=3.8
        conda activate Conv
        pip install pip --upgrade
        pip install -r requirements.txt
        conda activate Conv
        conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.1 -c pytorch -c nvidia
    ```

# Src 
--coef.py [the coeffients for calculation] 

--config.py [the configuration]

--dat.py [load the data]

--hook.py [the hook on the model to register data in the propagation process]

--models.py [the models used for the exps]

--train.py [train the models]

--utils.py [useful tools]

# Scripts
--bottleneck1.py [train the model to verify the bottleneck 1]

--bottleneck2.py [train the model to verify the bottleneck 2]

--corollary1.py [verify the corollary 1]

--corollary2.py [verify the corollary 2]

--plot_bottleneck1.py [visualize the bottleneck 1]

--plot_bottleneck2.py [visualize the bottleneck 2]

--plot_corollary.py [visualize the corollaries]

--remark1.py [verify the remark 1]

--remark2.py [verify the remark 2]

--remark3.py [verify the remark 3]

--remark4.py [verify the remark 4]

--remark5.py [verify the remark 5]

--theorem5.py [verify the theorem 5]

--theorem6.py [verify the theorem 6]




## Contact

Please contact [tling@sjtu.edu.cn] if you have any question on the codes.
    
---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[tling@sjtu.edu.cn]
