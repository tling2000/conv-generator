from datetime import datetime
import os
from pickle import TRUE

DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)  
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2) + \
    str(datetime.now().second).zfill(2)

KERNEL_SIZE = 3
IMAGE_SHAPE = (64,64)

IN_CHANNELS = 1
MID_CHANNELS = 8

CONV_NUM = 50

PARAM_MEAN = 0.01
PARAM_STD = 0.1

WITH_BIAS = False