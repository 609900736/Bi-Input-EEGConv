#coding:utf-8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.train import train_EEGNet, train_rawEEGConvNet, train_graphEEGConvNet, train_MB3DCNN, crossValidate, test
from tensorflow.python.keras import backend as K

srate = 250
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'CPU':10},
#                                               intra_op_parallelism_threads=10,
#                                               inter_op_parallelism_threads=2)))
K.set_image_data_format('channels_last')
tf.enable_eager_execution()

if __name__ == '__main__':
    # train_EEGNet(4,
    #              srate=srate,
    #              epochs=1200,
    #              patience=300,
    #              prep=False,
    #              beg=0,
    #              end=4)
    # train_rawEEGConvNet(4,
    #                     srate=srate,
    #                     epochs=1200,
    #                     patience=300,
    #                     prep=False,
    #                     beg=0,
    #                     end=4)
    # train_graphEEGConvNet(4,
    #                       srate=srate,
    #                       epochs=300,
    #                       patience=100,
    #                       prep=False,
    #                       beg=0,
    #                       end=4,
    #                       H=12,
    #                       W=14)
    # train_MB3DCNN(4,
    #               srate=srate,
    #               epochs=300,
    #               patience=100,
    #               prep=False,
    #               beg=0,
    #               end=1.25,
    #               H=6,
    #               W=7)
    crossValidate(train_EEGNet, K=10)(4,
                                      srate=srate,
                                      epochs=300,
                                      patience=100,
                                      prep=False,
                                      beg=0,
                                      end=4)
    pass