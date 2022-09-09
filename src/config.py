from datetime import datetime
import os

DATE = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)  
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2) + \
    str(datetime.now().second).zfill(2)
SRC_PATH = os.path.dirname(os.path.abspath(__file__))


KERNEL_SIZE = 3
IMAGE_SHAPE = (64,64)

MID_DIM = 5

SAMPLE_NUM = 1000
CONV_NUM = 5

PARAM_MEAN = 0.1
PARAM_STD = 0.1