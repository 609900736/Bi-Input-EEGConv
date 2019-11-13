#coding:utf-8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.train import create_EEGNet, create_rawEEGConvNet, create_graphEEGConvNet, create_MB3DCNN, crossValidate
from core.generators import rawGenerator, graphGenerator
from tensorflow.python.keras import backend as K

# tf.compat.v1.enable_eager_execution()
#K.set_session(tf.Session(config=tf.ConfigProto(device_count={'CPU':10},
#                                               intra_op_parallelism_threads=10,
#                                               inter_op_parallelism_threads=2)))
K.set_image_data_format('channels_last')
srate = 250


if __name__ == '__main__':
    crossValidate(create_rawEEGConvNet,
                  rawGenerator,
                  kFold=10,
                  beg=0,
                  end=4,
                  srate=srate,
                  epochs=1200,
                  patience=300)(4)
    pass