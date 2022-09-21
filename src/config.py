from datetime import datetime
import os

DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)  
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2) + \
    str(datetime.now().second).zfill(2)

KERNEL_SIZE = 7
IMAGE_SHAPE = (32,32)

IN_CHANNELS = 3
MID_CHANNELS = 16

CONV_NUM = 5

PARAM_MEAN = 0.01
PARAM_STD = 0.1

WITH_BIAS = False
WITH_UPSAMPLE = True