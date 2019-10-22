#coding:utf-8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, filterbank
from core.models import EEGNet
from core.train import train_EEGNet, train_rawEEGConvNet, train_graphEEGConvNet
from tensorflow.python.keras import backend as K

srate = 250
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'CPU':10},
#                                               intra_op_parallelism_threads=10,
#                                               inter_op_parallelism_threads=2)))
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    # train_EEGNet(4,
    #              srate=srate,
    #              epochs=1200,
    #              patience=300,
    #              prep=False,
    #              beg=0,
    #              end=4)
    train_rawEEGConvNet(4,
                        srate=srate,
                        epochs=1200,
                        patience=300,
                        prep=False,
                        beg=0,
                        end=4)
    # train_graphEEGConvNet(4,
    #                       srate=srate,
    #                       epochs=300,
    #                       patience=100,
    #                       prep=False,
    #                       beg=0,
    #                       end=4)
    pass