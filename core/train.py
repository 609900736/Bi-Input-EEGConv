# coding:utf-8

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, load_or_gen_filterbank_data, load_locs, load_or_gen_interestingband_data, load_or_generate_images, highpassfilter, bandpassfilter
from core.models import EEGNet, rawEEGConvModel, rawEEGConvNet, graphEEGConvModel, graphEEGConvNet, BiInputsEEGConvNet, ShallowConvNet, DeepConvNet, MB3DCNN
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model


def train_MB3DCNN(nClasses,
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
                  restate=True,
                  prep=False,
                  mode='raw',
                  averageImages=1):
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model = MB3DCNN(nClasses, H=H, W=W, Samples=Samples)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'MB3DCNN.png', show_shapes=True)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=patience,
                                                     verbose=0,
                                                     mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_or_generate_images(filepath,
                                          beg=beg,
                                          end=end,
                                          srate=srate,
                                          mode=mode,
                                          averageImages=averageImages,
                                          H=H,
                                          W=W)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_or_generate_images(filepath,
                                         beg=beg,
                                         end=end,
                                         srate=srate,
                                         mode=mode,
                                         averageImages=averageImages,
                                         H=H,
                                         W=W)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + '_MB3DCNN.h5')

        checkpointer = ModelCheckpoint(filepath=filepath,
                                       verbose=1,
                                       save_best_only=True)

        history.append(
            model.fit(x=x_train,
                      y=[y_train],
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[checkpointer, earlystopping],
                      verbose=verbose,
                      validation_data=[x_test, [y_test]]).history)

        if restate:
            model.reset_states()

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_MB3DCNN.npy')
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


def train_EEGNet(nClasses,
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
                 restate=True,
                 prep=False):
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

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=patience,
                                                     verbose=0,
                                                     mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_data(filepath, label=False)
        x_train = bandpassfilter(x_train)
        x_train = x_train[:, :,
                          math.floor(beg * srate):math.ceil(end *
                                                            srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_data(filepath, label=False)
        x_test = bandpassfilter(x_test)
        x_test = x_test[:, :,
                        math.floor(beg * srate):math.ceil(end *
                                                          srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_EEGNet.h5')

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

        if restate:
            model.reset_states()

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_EEGNet.npy')
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


def train_rawEEGConvNet(nClasses,
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
                        restate=True,
                        prep=False):
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

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=patience,
                                                     verbose=0,
                                                     mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_data(filepath, label=False)
        x_train = bandpassfilter(x_train)
        x_train = x_train[:, :,
                          math.floor(beg * srate):math.ceil(end *
                                                            srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_data(filepath, label=False)
        x_test = bandpassfilter(x_test)
        x_test = x_test[:, :,
                        math.floor(beg * srate):math.ceil(end *
                                                          srate), np.newaxis]
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_rawEEGConvNet.h5')

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

        if restate:
            model.reset_states()

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_rawEEGConvNet.npy')
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


def train_graphEEGConvNet(nClasses,
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
                          restate=True,
                          prep=True,
                          mode='topography',
                          averageImages=1):
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

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=patience,
                                                     verbose=0,
                                                     mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_or_generate_images(filepath,
                                          beg=beg,
                                          end=end,
                                          srate=srate,
                                          mode=mode,
                                          averageImages=averageImages,
                                          H=H,
                                          W=W)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_or_generate_images(filepath,
                                         beg=beg,
                                         end=end,
                                         srate=srate,
                                         mode=mode,
                                         averageImages=averageImages,
                                         H=H,
                                         W=W)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_graphEEGConvNet.h5')

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

        if restate:
            model.reset_states()

    filepath = os.path.join(
        'model',
        str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) + '_' +
        str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' + str(tm.tm_sec) +
        '_graphEEGConvNet.npy')
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


def train_BiInputsEEGConvNet(nClasses,
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
                             restate=True,
                             prep=True,
                             mode='potential'):
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

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0,
                                                     patience=patience,
                                                     verbose=0,
                                                     mode='auto')

    tm = time.localtime()
    history = []
    if not os.path.exists('model'):  # 判断是否存在
        os.makedirs('model')  # 不存在则创建
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_or_gen_filterbank_data(filepath,
                                              beg=beg,
                                              end=end,
                                              srate=srate)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_or_gen_filterbank_data(filepath,
                                             beg=beg,
                                             end=end,
                                             srate=srate)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_rawEEGConvNet.h5')

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

        if restate:
            net_s.reset_states()

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
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_or_generate_images(filepath,
                                          beg=beg,
                                          end=end,
                                          srate=srate,
                                          mode=mode,
                                          averageImages=64)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_or_generate_images(filepath,
                                         beg=beg,
                                         end=end,
                                         srate=srate,
                                         mode=mode,
                                         averageImages=64)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_graphEEGConvNet.h5')

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

        if restate:
            net_g.reset_states()

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
    for i in range(1, 10):
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T' + pp + '.mat')
        x_train = load_or_gen_filterbank_data(filepath,
                                              beg=beg,
                                              end=end,
                                              srate=srate)
        filepath = os.path.join('data', dataSelect, 'Train',
                                'A0' + str(i) + 'T_label' + pp + '.mat')
        y_train = load_data(filepath)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_or_gen_filterbank_data(filepath,
                                             beg=beg,
                                             end=end,
                                             srate=srate)
        filepath = os.path.join('data', dataSelect, 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath)

        filepath = os.path.join(
            'model',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_A0' + str(i) + 'T_BiInputsEEGConvNet.h5')

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

        if restate:
            model.reset_states()

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