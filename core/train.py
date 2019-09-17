#coding:utf-8

import os
import numpy as np
import math as m
import tensorflow as tf
import time
import scipy.io as sio
import pandas as pd
from tensorflow.python.keras import optimizers as opt
from tensorflow.python.keras.callbacks import ModelCheckpoint

import core.models as models
import core.utils as utils

def make_checkpoint():
    filepath = os.path.join('./model','checkpoint.h5')
    if not os.path.exists('./model'): #判断是否存在
        os.makedirs('./model') #不存在则创建
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                   save_best_only=True)
    return filepath, checkpointer

def train(data, label, model='BIEEGConvNet',):

    return

