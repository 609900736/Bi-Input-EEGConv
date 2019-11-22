#coding:utf-8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.train import create_EEGNet, create_rawEEGConvNet, create_graphEEGConvNet, create_MB3DCNN, crossValidate
from core.generators import rawGenerator, graphGenerator
from core.splits import StratifiedKFold, AllTrain

from tensorflow_core.python.keras import backend as K


K.set_image_data_format('channels_last')
srate = 250


if __name__ == '__main__':
    crossValidate(create_rawEEGConvNet,
                  rawGenerator,
                  AllTrain,
                  kFold=5,
                  beg=0,
                  end=4,
                  srate=srate,
                  epochs=1200,
                  patience=300)(4, F=16, D=10, Ns=20, tl1=1e-5)
    pass