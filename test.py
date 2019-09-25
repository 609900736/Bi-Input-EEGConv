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

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

srate = 250
start = 0
end = 4
prep = True
Samples = (end-start)*srate
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    if prep:
        pp = '_pp'
    else:
        pp = ''

    for i in range(1,10):
        filepath = os.path.join('./data/Test','A0'+str(i)+'E'+pp+'.mat')
        x_test.append(load_data(filepath,label=False))
        x_test[-1] = np.expand_dims(x_test[-1][:,:,start*srate:end*srate],-1)
        filepath = os.path.join('./data/Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test.append(load_data(filepath,label=True))
        y_test[-1] -= 1

    for i in range(1,10):
        filepath = os.path.join('./model/2019_9_25_14_0_18'+'_A0'+str(i)+
                                'T_EEGNet.h5')
        model = load_model(filepath,compile=False)
        #model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        #          loss=tf.keras.losses.sparse_categorical_crossentropy,
        #          metrics=['accuracy'])
        x = x_test.pop(0)
        y = y_test.pop(0)
        model.evaluate(x,y,batch_size=10,verbose=2)

    pass