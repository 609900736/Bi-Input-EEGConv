#coding:utf-8
import core.models as models
import core.train as train
import core.utils as utils
import scipy.io as sio

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, filterbank
from core.models import EEGNet
from core.train import train, make_checkpoint
from tensorflow.python.keras import backend as K

srate = 250
K.set_image_data_format('channels_first')

if __name__ == '__main__':
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(1,10):
        filepath = os.path.join('./data/Train','A0'+str(i)+'T_pp.mat')
        x_train.append(load_data(filepath,label=False))
        x_train[-1] = np.expand_dims(x_train[-1][:,:,125:625],1)
        filepath = os.path.join('./data/Train','A0'+str(i)+'T_label_pp.mat')
        y_train.append(load_data(filepath,label=True))
        y_train[-1] -= 1
        filepath = os.path.join('./data/Test','A0'+str(i)+'E_pp.mat')
        x_test.append(load_data(filepath,label=False))
        x_test[-1] = np.expand_dims(x_test[-1][:,:,125:625],1)
        filepath = os.path.join('./data/Test','A0'+str(i)+'E_label_pp.mat')
        y_test.append(load_data(filepath,label=True))
        y_test[-1] -= 1
        
    model = EEGNet(4,Chans=22,Samples=500,kernLength=64)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3,amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    for i in range(1,10):
        filepath = os.path.join('./model/2019_9_24_10_4_13'+'_A0'+str(i)+
                                'T_EEGNet.h5')
        model.load_weights(filepath)
        x = x_test.pop(0)
        y = y_test.pop(0)
        model.evaluate(x,y,batch_size=10,verbose=2)
        #model.reset_states()

    pass