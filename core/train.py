# coding:utf-8

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, load_or_gen_filterbank_data, load_locs, load_or_gen_interestingband_data, load_or_generate_images, highpassfilter, bandpassfilter
from core.models import EEGNet, rawEEGConvModel, rawEEGConvNet, graphEEGConvModel, graphEEGConvNet, BiInputsEEGConvNet, ShallowConvNet, DeepConvNet, MB3DCNN
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate


def train_MB3DCNN(nClasses,
                  subject,
                  H,
                  W,
                  beg=0,
                  end=4,
                  srate=250,
                  dataSelect='4s',
                  batch_size=10,
                  epochs=500,
                  verbose=2,
                  patience=100,
                  drawflag=False,
                  prep=False,
                  mode='raw',
                  averageImages=1,
                  data=None):
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    tm = time.localtime()
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    if data is None:
        data = {
            'x_train': None,
            'x_test': None,
            'y_train': None,
            'y_test': None
        }
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T' + pp + '.mat')
        data['x_train'] = load_or_generate_images(filepath,
                                                  beg=beg,
                                                  end=end,
                                                  srate=srate,
                                                  mode=mode,
                                                  averageImages=averageImages,
                                                  H=H,
                                                  W=W)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T_label' + pp + '.mat')
        data['y_train'] = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E' + pp + '.mat')
        data['x_test'] = load_or_generate_images(filepath,
                                                 beg=beg,
                                                 end=end,
                                                 srate=srate,
                                                 mode=mode,
                                                 averageImages=averageImages,
                                                 H=H,
                                                 W=W)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E_label' + pp + '.mat')
        data['y_test'] = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + '_MB3DCNN.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto')

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  verbose=0,
                                  mode='auto')

    # model = MB3DCNN(nClasses, H=H, W=W, Samples=Samples)
    model = KerasClassifier(built_fn=MB3DCNN,
                            nClasses=nClasses,
                            H=H,
                            W=W,
                            Samples=Samples)
    kfold = StratifiedKFold(n_splits=10)
    results = cross_val_score(
        model,
        data['x_train'], [data['y_train']],
        cv=kfold,
        fit_params=dict(batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, earlystopping],
                        verbose=verbose,
                        validation_data=[data['x_test'], [data['y_test']]]))

    # history = model.fit(x=data['x_train'],
    #                     y=[data['y_train']],
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     callbacks=[checkpointer, earlystopping],
    #                     verbose=verbose,
    #                     validation_data=[data['x_test'],
    #                                      [data['y_test']]]).history

    model = load_model(filepath)
    loss, acc = model.evaluate(data['x_test'], [data['y_test']],
                               batch_size=batch_size,
                               verbose=verbose)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_MB3DCNN.npy')
    np.save(filepath, history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
        for i in range(1, 10):
            # Plot training & validation accuracy values
            plt.figure(2 * i - 1)
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.figure(2 * i)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    del model, history
    return data, acc


def train_EEGNet(nClasses,
                 subject,
                 Chans=22,
                 beg=0,
                 end=4,
                 srate=250,
                 dataSelect='4s',
                 batch_size=10,
                 epochs=500,
                 verbose=2,
                 patience=100,
                 drawflag=False,
                 prep=False,
                 data=None):
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model = EEGNet(nClasses, Chans=Chans, Samples=Samples)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    # filepath = os.path.join('model', '2019_10_10_17_8_29_A09T_EEGNet.h5')
    # model = load_model(filepath)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'EEGNet.png', show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  verbose=0,
                                  mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    if data is None:
        data = {
            'x_train': None,
            'x_test': None,
            'y_train': None,
            'y_test': None
        }
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T' + pp + '.mat')
        data['x_train'] = load_data(filepath, label=False)
        data['x_train'] = bandpassfilter(data['x_train'])
        data['x_train'] = data['x_train'][:, :,
                                          math.floor(beg * srate):math.
                                          ceil(end * srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T_label' + pp + '.mat')
        data['y_train'] = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E' + pp + '.mat')
        data['x_test'] = load_data(filepath, label=False)
        data['x_test'] = bandpassfilter(data['x_test'])
        data['x_test'] = data['x_test'][:, :,
                                        math.floor(beg * srate):math.
                                        ceil(end * srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E_label' + pp + '.mat')
        data['y_test'] = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_EEGNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history = model.fit(x=data['x_train'],
                        y=data['y_train'],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, earlystopping],
                        verbose=verbose,
                        validation_data=[data['x_test'],
                                         data['y_test']]).history

    model = load_model(filepath)
    loss, acc = model.evaluate(data['x_test'],
                               data['y_test'],
                               batch_size=batch_size,
                               verbose=verbose)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_EEGNet.npy')
    np.save(filepath, history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
        for i in range(1, 10):
            # Plot training & validation accuracy values
            plt.figure(2 * i - 1)
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.figure(2 * i)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    del model, history
    return data, acc


def train_rawEEGConvNet(nClasses,
                        subject,
                        Chans=22,
                        beg=0,
                        end=4,
                        Colors=1,
                        srate=250,
                        dataSelect='4s',
                        batch_size=10,
                        epochs=500,
                        verbose=2,
                        patience=100,
                        drawflag=False,
                        prep=False,
                        data=None):
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model = rawEEGConvModel(Chans=Chans, Samples=Samples, Colors=Colors)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'rawEEGConvModel.png', show_shapes=True)

    model = rawEEGConvNet(nClasses,
                          model,
                          Chans=Chans,
                          Samples=Samples,
                          Colors=Colors)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'rawEEGConvNet.png', show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  verbose=0,
                                  mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    if data is None:
        data = {
            'x_train': None,
            'x_test': None,
            'y_train': None,
            'y_test': None
        }
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T' + pp + '.mat')
        data['x_train'] = load_data(filepath, label=False)
        data['x_train'] = bandpassfilter(data['x_train'])
        data['x_train'] = data['x_train'][:, :,
                                          math.floor(beg * srate):math.
                                          ceil(end * srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T_label' + pp + '.mat')
        data['y_train'] = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E' + pp + '.mat')
        data['x_test'] = load_data(filepath, label=False)
        data['x_test'] = bandpassfilter(data['x_test'])
        data['x_test'] = data['x_test'][:, :,
                                        math.floor(beg * srate):math.
                                        ceil(end * srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E_label' + pp + '.mat')
        data['y_test'] = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_rawEEGConvNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history = model.fit(x=data['x_train'],
                        y=data['y_train'],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, earlystopping],
                        verbose=verbose,
                        validation_data=[data['x_test'],
                                         data['y_test']]).history

    model = load_model(filepath)
    loss, acc = model.evaluate(data['x_test'],
                               data['y_test'],
                               batch_size=batch_size,
                               verbose=verbose)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_rawEEGConvNet.npy')
    np.save(filepath, history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
        for i in range(1, 10):
            # Plot training & validation accuracy values
            plt.figure(2 * i - 1)
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.figure(2 * i)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    del model, history
    return data, acc


def train_graphEEGConvNet(nClasses,
                          subject,
                          Colors=1,
                          H=30,
                          W=35,
                          beg=0,
                          end=4,
                          srate=250,
                          dataSelect='4s',
                          batch_size=10,
                          epochs=500,
                          verbose=2,
                          patience=100,
                          drawflag=False,
                          prep=True,
                          mode='topography',
                          averageImages=1,
                          data=None):
    Samples = math.ceil(end * srate - beg * srate) // averageImages
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model = graphEEGConvModel(Colors=Colors, Samples=Samples, H=H, W=W)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'graphEEGConvModel.png', show_shapes=True)

    model = graphEEGConvNet(nClasses,
                            model,
                            Colors=Colors,
                            Samples=Samples,
                            H=H,
                            W=W)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'graphEEGConvNet.png', show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  verbose=0,
                                  mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    if data is None:
        data = {
            'x_train': None,
            'x_test': None,
            'y_train': None,
            'y_test': None
        }
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T' + pp + '.mat')
        data['x_train'] = load_or_generate_images(filepath,
                                                  beg=beg,
                                                  end=end,
                                                  srate=srate,
                                                  mode=mode,
                                                  averageImages=averageImages,
                                                  H=H,
                                                  W=W)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(subject) + 'T_label' + pp + '.mat')
        data['y_train'] = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E' + pp + '.mat')
        data['x_test'] = load_or_generate_images(filepath,
                                                 beg=beg,
                                                 end=end,
                                                 srate=srate,
                                                 mode=mode,
                                                 averageImages=averageImages,
                                                 H=H,
                                                 W=W)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(subject) + 'E_label' + pp + '.mat')
        data['y_test'] = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_graphEEGConvNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history = model.fit(x=data['x_train'],
                        y=data['y_train'],
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpointer, earlystopping],
                        verbose=verbose,
                        validation_data=[data['x_test'],
                                         data['y_test']]).history

    model = load_model(filepath)
    loss, acc = model.evaluate(data['x_test'],
                               data['y_test'],
                               batch_size=batch_size,
                               verbose=verbose)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_graphEEGConvNet.npy')
    np.save(filepath, history)
    #history = np.load(filepath,allow_pickle=True)
    if drawflag:
        for i in range(1, 10):
            # Plot training & validation accuracy values
            plt.figure(2 * i - 1)
            plt.plot(history['acc'])
            plt.plot(history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.figure(2 * i)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    del model, history
    return data, acc


def train_BiInputsEEGConvNet(nClasses,
                             subject,
                             Colors=8,
                             Chans=22,
                             W=16,
                             H=16,
                             beg=0,
                             end=4,
                             srate=250,
                             dataSelect='4s',
                             batch_size=10,
                             epochs=500,
                             verbose=2,
                             patience=100,
                             drawflag=False,
                             prep=True,
                             mode='topography',
                             averageImages=1,
                             data=None):
    '''
    Don't use this, it need to be restructed
    '''
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model_s = rawEEGConvModel(Colors=Colors, Chans=Chans, Samples=Samples)
    model_s.summary()
    # export graph of the model_s
    tf.keras.utils.plot_model(model_s, 'rawEEGConvModel.png', show_shapes=True)

    net_s = rawEEGConvNet(nClasses,
                          model_s,
                          Colors=Colors,
                          Chans=Chans,
                          Samples=Samples)

    net_s.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    net_s.summary()
    tf.keras.utils.plot_model(net_s, 'rawEEGConvNet.png', show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=patience,
                                  verbose=0,
                                  mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建

    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T' + pp + '.mat')
    x_train = load_or_gen_filterbank_data(filepath,
                                          beg=beg,
                                          end=end,
                                          srate=srate)
    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T_label' + pp + '.mat')
    y_train = load_data(filepath)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E' + pp + '.mat')
    x_test = load_or_gen_filterbank_data(filepath,
                                         beg=beg,
                                         end=end,
                                         srate=srate)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E_label' + pp + '.mat')
    y_test = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_rawEEGConvNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history.append(
        net_s.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[checkpointer, earlystopping],
                  verbose=verbose,
                  validation_data=[x_test, y_test]).history)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_rawEEGConvNet.npy')
    np.save(filepath, history)

    model_g = graphEEGConvModel(Colors=Colors, Samples=Samples, H=H, W=W)
    model_g.summary()
    # export graph of the model_g
    tf.keras.utils.plot_model(model_g,
                              'graphEEGConvModel.png',
                              show_shapes=True)

    net_g = graphEEGConvNet(nClasses,
                            model_g,
                            Colors=Colors,
                            Samples=Samples,
                            H=H,
                            W=W)
    net_g.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    net_g.summary()
    tf.keras.utils.plot_model(net_g, 'graphEEGConvNet.png', show_shapes=True)

    history = []

    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T' + pp + '.mat')
    x_train = load_or_generate_images(filepath,
                                      beg=beg,
                                      end=end,
                                      srate=srate,
                                      mode=mode,
                                      averageImages=64)
    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T_label' + pp + '.mat')
    y_train = load_data(filepath)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E' + pp + '.mat')
    x_test = load_or_generate_images(filepath,
                                     beg=beg,
                                     end=end,
                                     srate=srate,
                                     mode=mode,
                                     averageImages=64)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E_label' + pp + '.mat')
    y_test = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_graphEEGConvNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history.append(
        net_g.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[checkpointer, earlystopping],
                  verbose=verbose,
                  validation_data=[x_test, y_test]).history)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_graphEEGConvNet.npy')
    np.save(filepath, history)

    model = BiInputsEEGConvNet(4,
                               model_s,
                               model_g,
                               Chans=Chans,
                               Samples=Samples,
                               Colors=Colors,
                               H=H,
                               W=W)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, amsgrad=True),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model,
                              'BiInputsEEGConvNet.png',
                              show_shapes=True)

    history = []

    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T' + pp + '.mat')
    x_train = load_or_gen_filterbank_data(filepath,
                                          beg=beg,
                                          end=end,
                                          srate=srate)
    filepath = os.path.join('data', dataSelect, 'Train',
                            'A0' + str(subject) + 'T_label' + pp + '.mat')
    y_train = load_data(filepath)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E' + pp + '.mat')
    x_test = load_or_gen_filterbank_data(filepath,
                                         beg=beg,
                                         end=end,
                                         srate=srate)
    filepath = os.path.join('data', dataSelect, 'Test',
                            'A0' + str(subject) + 'E_label' + pp + '.mat')
    y_test = load_data(filepath)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' +
        str(subject) + 'T_BiInputsEEGConvNet.h5')

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=True)

    history.append(
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[checkpointer, earlystopping],
                  verbose=verbose,
                  validation_data=[x_test, y_test]).history)

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_BiInputsEEGConvNet.npy')
    np.save(filepath, history)

    #history = np.load(filepath,allow_pickle=True)

    if drawflag:
        for i in range(1, 10):
            h = history.pop(0)

            # Plot training & validation accuracy values
            plt.figure(2 * i - 1)
            plt.plot(h['acc'])
            plt.plot(h['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.figure(2 * i)
            plt.plot(h['loss'])
            plt.plot(h['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

        plt.show()


class crossValidate(object):
    '''
    Class for K-fold Cross Validation.

    This class has implemented a magic function, for which it can be used like a function:
    ```python
    def func(*args, **kwargs, subject=None, data=None):
        ...
        return data, acc
    ...
    avg_acc = crossValidate(func, K=10, num=9)(*args, **kwargs)
    ```
    Input:
    ```txt
    func        : function, Function needs to do cross validation.
    splitFunc   : function, Support class from sklearn.model_selection.
    kFold       : int, Number of K-fold.
    shuffle     : bool, optional Whether to shuffle each class's samples before 
                  splitting into batches.
    random_state: int, RandomState instance or None, optional, default = None. If int, 
                  random_state is the seed used by the random number generator; If 
                  RandomState instance, random_state is the random number generator; 
                  If None, the random number generator is the RandomState instance 
                  used by np.random. Used when shuffle == True.
    num         : int, Number of subjects.
    ```
    Output:
    ```txt
    avg_acc     : list, Average accuracy for each subject with K-fold Cross Validation, 
                  and total average accuracy in the last of list
    ```
    More details to see the codes.
    '''
    def __init__(self,
                 func,
                 splitFunc=StratifiedKFold,
                 kFold=10,
                 shuffle=False,
                 random_state=None,
                 num=9):
        self.func = func
        self.splitFunc = splitFunc
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.num = num + 1

    def __call__(self, *args, **kwargs):
        avg_acc = []
        for i in range(1, self.num):
            data = None
            accik = []
            for k in range(self.kFold):
                data, acc = self.func(*args, **kwargs, subject=i, data=data)
                accik.append(acc)
            avg_acc.append(np.average(np.asarray(accik)))
            del data
        total_avg_acc = np.average(np.asarray(avg_acc))
        print('{:d}-fold Cross Validation Accuracy'.format(self.kFold))
        for i in range(1, self.num):
            print('Subject {0:0>2d}: {1:.2%}'.format(i, avg_acc[i - 1]))
        print('Average   : {:.2%}'.format(total_avg_acc))
        avg_acc.append(total_avg_acc)
        return avg_acc

    def getConfig(self):
        print('Method: {0:s}\nCross Validation Fold: {1:d}'.format(
            self.func.__name__, self.kFold))

    def setConfig(self,
                  func,
                  splitFunc=StratifiedKFold,
                  kFold=10,
                  shuffle=False,
                  random_state=None,
                  num=9):
        self.func = func
        self.splitFunc = splitFunc
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.num = num + 1

    def _read_data(self, filepath):
        
        pass

    def _spilt(self, X, y):
        skf = self.splitFunc(n_splits=self.kFold,
                             shuffle=self.shuffle,
                             random_state=self.random_state)
        for train_index, test_index in skf.split(X, y):
            # (x_train, y_train), (x_test, y_test)
            yield (X[train_index], y[train_index]), (X[test_index],
                                                     y[test_index])


def test(*args, **kwargs):
    print(args, kwargs)
    return None, 1.
