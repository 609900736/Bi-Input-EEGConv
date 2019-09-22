#coding:utf-8
import core.models as models
import core.train as train
import core.utils as utils
import scipy.io as sio

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data
from core.models import EEGNet
from core.train import train, make_checkpoint
from tensorflow.python.keras import backend as K

srate = 250
K.set_session(tf.Session(config=tf.ConfigProto(device_count={'CPU':10},
                                               intra_op_parallelism_threads=10,
                                               inter_op_parallelism_threads=2)))

if __name__=='__main__':
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(1,10):
        filepath = os.path.join('./data/Train','A0'+str(i)+'T_pp.mat')
        x_train.append(load_data(filepath,label=False))
        x_train[-1] = np.expand_dims(x_train[-1][:,:22,:],1)
        filepath = os.path.join('./data/Train','A0'+str(i)+'T_label_pp.mat')
        y_train.append(load_data(filepath,label=True))
        y_train[-1] -= 1
        filepath = os.path.join('./data/Test','A0'+str(i)+'E_pp.mat')
        x_test.append(load_data(filepath,label=False))
        x_test[-1] = np.expand_dims(x_test[-1][:,:22,:],1)
        filepath = os.path.join('./data/Test','A0'+str(i)+'E_label_pp.mat')
        y_test.append(load_data(filepath,label=True))
        y_test[-1] -= 1
        
    model = EEGNet(4,Chans=22,Samples=1000,kernLength=64)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3,amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # 保存模型图
    tf.keras.utils.plot_model(model, 'EEGNet.png')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                     patience=50, verbose=0, mode='auto')

    tm = time.localtime()
    history = []
    for i in range(1,10):
        filepath = os.path.join('./model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_EEGNet.h5')
        checkpointer = make_checkpoint(filepath)
        history.append(model.fit(x=x_train.pop(0),y=y_train.pop(0),batch_size=10,
                                 epochs=500,callbacks=[checkpointer,earlystopping],
                                 verbose=2,validation_data=[x_test.pop(0),y_test.pop(0)]).history)
        model.reset_states()

    filepath = os.path.join('./model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_EEGNet.npy')
    np.save(filepath,history)
    #history = np.load(filepath,allow_pickle=True)
    for i in range(1,10):
        h = history.pop(0)
        
        # Plot training & validation accuracy values
        plt.figure(2*i-1)
        plt.plot(h['acc'])
        plt.plot(h['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.figure(2*i)
        plt.plot(h['loss'])
        plt.plot(h['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
    pass