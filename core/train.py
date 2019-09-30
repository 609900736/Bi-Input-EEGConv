#coding:utf-8

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, load_or_gen_filterbank_data, load_locs, load_or_gen_interestingband_data
from core.models import EEGNet, rawEEGConvModel, rawEEGConvNet, graphEEGConvModel, graphEEGConvNet, BiInputsEEGConvNet
from tensorflow.python.keras.callbacks import ModelCheckpoint


def train_EEGNet(n_classes, Chans=22, start=0, end=3, srate=250, 
                 batch_size=10, epochs=500, verbose=2, patience=100, 
                 drawflag=False, restate=True, prep=False):
    Samples = (end-start)*srate
    if prep:
        pp = '_pp'
    else:
        pp = ''
        
    model = EEGNet(n_classes,Chans=Chans,Samples=Samples)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    tf.keras.utils.plot_model(model, 'EEGNet.png', show_shapes=True)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                     patience=patience, verbose=0, mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'): # 判断是否存在
            os.makedirs('model') # 不存在则创建
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_data(filepath,label=False)
        x_train = np.expand_dims(x_train[:,:,start*srate:end*srate],-1)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_data(filepath,label=False)
        x_test = np.expand_dims(x_test[:,:,start*srate:end*srate],-1)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_EEGNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(model.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_EEGNet.npy')
    np.save(filepath,history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
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


def train_rawEEGConvNet(n_classes, Chans=22, start=0, end=3, Colors=1,
                        srate=250, batch_size=10, epochs=500, verbose=2, 
                        patience=100, drawflag=False, restate=True, 
                        prep=False):
    Samples = (end-start)*srate
    if prep:
        pp = '_pp'
    else:
        pp = ''
        
    model = rawEEGConvModel(Chans=Chans,Samples=Samples,Colors=Colors)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'rawEEGConvModel.png', show_shapes=True)

    model = rawEEGConvNet(n_classes,model,Chans=Chans,Samples=Samples,Colors=Colors)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'rawEEGConvNet.png', show_shapes=True)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                     patience=patience, verbose=0, mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'): # 判断是否存在
            os.makedirs('model') # 不存在则创建
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_or_gen_interestingband_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_or_gen_interestingband_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_rawEEGConvNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(model.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_rawEEGConvNet.npy')
    np.save(filepath,history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
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


def train_graphEEGConvNet(n_classes, Colors=8, Chans=22, W=16, H=16, 
                          start=0, end=4, srate=250, batch_size=10, 
                          epochs=500, verbose=2, patience=100, 
                          drawflag=False, restate=True, prep=True):
    Samples = (end-start)*srate
    if prep:
        pp = '_pp'
    else:
        pp = ''
       
    model = graphEEGConvModel(Colors=Colors,Samples=Samples,H=H,W=W)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'graphEEGConvModel.png', show_shapes=True)

    model = graphEEGConvNet(n_classes,model,Colors=Colors,Samples=Samples,H=H,W=W)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'graphEEGConvNet.png', show_shapes=True)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                     patience=patience, verbose=0, mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'): # 判断是否存在
            os.makedirs('model') # 不存在则创建
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_graphEEGConvNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(model.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_graphEEGConvNet.npy')
    np.save(filepath,history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
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


def train_BiInputsEEGConvNet(n_classes, Colors=8, Chans=22, W=16, H=16, 
                             start=0, end=4, srate=250, batch_size=10, 
                             epochs=500, verbose=2, patience=100, 
                             drawflag=False, restate=True, prep=True):
    Samples = (end-start)*srate
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model_s = rawEEGConvModel(Colors=Colors,Chans=Chans,Samples=Samples)
    model_s.summary()
    # export graph of the model_s
    tf.keras.utils.plot_model(model_s, 'rawEEGConvModel.png', show_shapes=True)

    net_s = rawEEGConvNet(n_classes,model_s,Colors=Colors,Chans=Chans,Samples=Samples)

    net_s.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    net_s.summary()
    tf.keras.utils.plot_model(net_s, 'rawEEGConvNet.png', show_shapes=True)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
                                                     patience=patience, verbose=0, mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'): # 判断是否存在
            os.makedirs('model') # 不存在则创建
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_rawEEGConvNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(net_s.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_rawEEGConvNet.npy')
    np.save(filepath,history)
    
    model_g = graphEEGConvModel(Colors=Colors,Samples=Samples,H=H,W=W)
    model_g.summary()
    # export graph of the model_g
    tf.keras.utils.plot_model(model_g, 'graphEEGConvModel.png', show_shapes=True)

    net_g = graphEEGConvNet(n_classes,model_g,Colors=Colors,Samples=Samples,H=H,W=W)
    net_g.compile(optimizer=tf.keras.optimizers.Adam(1e-3,amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    net_g.summary()
    tf.keras.utils.plot_model(net_g, 'graphEEGConvNet.png', show_shapes=True)

    history = []
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_graphEEGConvNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(net_g.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_graphEEGConvNet.npy')
    np.save(filepath,history)

    model = BiInputsEEGConvNet(4,model_s,model_g,Chans=Chans,Samples=Samples,Colors=Colors,H=H,W=W)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3,amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'BiInputsEEGConvNet.png', show_shapes=True)

    history = []
    for i in range(1,10):
        filepath = os.path.join('data','Train','A0'+str(i)+'T'+pp+'.mat')
        x_train = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Train','A0'+str(i)+'T_label'+pp+'.mat')
        y_train = load_data(filepath)
        y_train -= 1
        filepath = os.path.join('data','Test','A0'+str(i)+'E'+pp+'.mat')
        x_test = load_or_gen_filterbank_data(filepath,start=start,end=end,srate=srate)
        filepath = os.path.join('data','Test','A0'+str(i)+'E_label'+pp+'.mat')
        y_test = load_data(filepath)
        y_test -= 1

        filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                                '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                                str(tm.tm_min)+'_'+str(tm.tm_sec)+'_A0'+str(i)+
                                'T_BiInputsEEGConvNet.h5')
        
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)

        history.append(model.fit(x=x_train,y=y_train,batch_size=batch_size,
                                 epochs=epochs,callbacks=[checkpointer,earlystopping],
                                 verbose=verbose,validation_data=[x_test,y_test]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join('model',str(tm.tm_year)+'_'+str(tm.tm_mon)+
                            '_'+str(tm.tm_mday)+'_'+str(tm.tm_hour)+'_'+
                            str(tm.tm_min)+'_'+str(tm.tm_sec)+'_BiInputsEEGConvNet.npy')
    np.save(filepath,history)

    #history = np.load(filepath,allow_pickle=True)

    if drawflag:
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


def _old_train(data, label, batch_size=1, model=None, optimizer='adam',
          loss='sparse_categorical_crossentropy', metrics=['accuracy'],
          epochs=300, validation_split=0.0, validation_data=None):
    if not (type(model) is tf.keras.Model):
        raise(ValueError('wrong type of model'))

    tm = time.localtime()
    filepath = os.path.join('./model',str(tm.tm_year)+'-'+str(tm.tm_mon)+
                            '-'+str(tm.tm_mday)+'-'+str(tm.tm_hour)+'-'+
                            str(tm.tm_min)+'-'+str(tm.tm_sec)+'-checkpoint.h5')
    checkpointer = make_checkpoint(filepath)

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    history = model.fit(x=data,y=label,batch_size=batch_size,epochs=epochs,
              callbacks=[checkpointer],validation_split=validation_split,
              validation_data=validation_data)

    model.load_weights(filepath)
    model.evaluate(data,label)

    filepath = os.path.join('./model',str(tm.tm_year)+'-'+str(tm.tm_mon)+
                            '-'+str(tm.tm_mday)+'-'+str(tm.tm_hour)+'-'+
                            str(tm.tm_min)+'-'+str(tm.tm_sec)+'-model.h5')
    model.save(filepath)

    # Plot training & validation accuracy values
    plt.figure(0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return model
