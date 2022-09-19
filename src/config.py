from datetime import datetime
import os
from pickle import TRUE

DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)  
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2) + \
    str(datetime.now().second).zfill(2)

KERNEL_SIZE = 3
IMAGE_SHAPE = (14,14)

IN_CHANNELS = 3
MID_CHANNELS = 16

CONV_NUM = 10

PARAM_MEAN = 0
PARAM_STD = 0.1

WITH_BIAS = False
WITH_UPSAMPLE = True