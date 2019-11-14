# coding:utf-8

import os
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model

from core.utils import load_data, load_or_gen_filterbank_data, load_locs, load_or_gen_interestingband_data, load_or_generate_images, highpassfilter, bandpassfilter
from core.models import EEGNet, rawEEGConvNet, graphEEGConvNet, BiInputsEEGConvNet, ShallowConvNet, DeepConvNet, MB3DCNN
from core.splits import StratifiedKFold


def create_MB3DCNN(nClasses,
                   H,
                   W,
                   Samples,
                   optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy']):
    model = MB3DCNN(nClasses, H=H, W=W, Samples=Samples)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'MB3DCNN.png', show_shapes=True)
    return model


def create_EEGNet(nClasses,
                  Samples,
                  Chans=22,
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']):
    model = EEGNet(nClasses, Chans=Chans, Samples=Samples)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'EEGNet.png', show_shapes=True)
    return model


def create_rawEEGConvNet(nClasses,
                         Samples,
                         Chans=22,
                         Colors=1,
                         optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy']):
    model = rawEEGConvNet(nClasses,
                          Chans=Chans,
                          Samples=Samples,
                          Colors=Colors)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'rawEEGConvNet.png', show_shapes=True)
    return model


def create_graphEEGConvNet(nClasses,
                           Samples,
                           Colors=1,
                           H=30,
                           W=35,
                           optimizer=tf.keras.optimizers.Adam(1e-3),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']):
    model = graphEEGConvNet(nClasses, Colors=Colors, Samples=Samples, H=H, W=W)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # export graph of the model
    model.summary()
    tf.keras.utils.plot_model(model, 'graphEEGConvNet.png', show_shapes=True)
    return model


def create_BiInputsEEGConvNet(nClasses,
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
    TODO: Don't use this, it need to be restructured.
    '''
    Samples = math.ceil(end * srate - beg * srate)
    if prep:
        pp = '_pp'
    else:
        pp = ''

    model_s = rawEEGConvNet(Colors=Colors, Chans=Chans, Samples=Samples)
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

    model_g = graphEEGConvNet(Colors=Colors, Samples=Samples, H=H, W=W)
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

    This framework can collect `model`, `loss`, `acc` and `history` from each fold, and 
    save them into files. 
    Data spliting methods from sklearn.model_selection are supported. you can pass the 
    classes as `splitMethod`. 

    This class has implemented a magic method, for which it can be used like a function.

    Parameters
    ----------
    ```txt
    built_in        : function, Create Training model which need to cross validate.
                      Please use `create_` at the begining of function name, 
                      like `create_model`.
    dataGent        : class, Generate data for @built_in, shapes (n_trails, ...). 
                      It should discriminate data and label.
                      More details see core.generators.
    splitMethod     : class, Support split methods from module sklearn.model_selection.
    kFold           : int, Number of K-fold.
    shuffle         : bool, Optional Whether to shuffle each class's samples before 
                      splitting into batches, default = False.
    random_state    : int, RandomState instance or None, optional, default = None. 
                      If int, random_state is the seed used by the random number 
                      generator; If RandomState instance, random_state is the random 
                      number generator; If None, the random number generator is the 
                      RandomState instance used by np.random. Used when shuffle == True.
    subs            : int, Number of subjects.
    isCropped       : bool, Switch of cropped training.
    batch_size      : int, Batch size.
    epochs          : int, Training epochs.
    patience        : int, Early stopping patience.
    verbose         : int, One of 0, 1 and 2.
    *a, *args       : tuple, Parameters used by @dataGent and @built_in respectively
    **kw, **kwargs  : dict, Parameters used by @dataGent and @built_in respectively, 
                      **kw should include parameters called `beg`, `end` and `srate`.
    ```

    Returns
    -------
    ```txt
    avg_acc         : list, Average accuracy for each subject with K-fold Cross Validation, 
                      and total average accuracy in the last of list
    ```

    Example
    -------
    ```python
    from core.splits import StratifiedKFold

    def create_model(Samples, *args, **kwargs):
        ...
        return acc

    class dataGenerator:
        def __init__(self, *a, **kw, beg=0, end=4, srate=250):
            ...

        def __call__(self, filepath, label=False):
            if label:
                ...
                return label
            else:
                ...
                return data
        ...
    ...
    avg_acc = crossValidate(
                model, 
                dataGenerator, 
                beg=0,
                end=4,
                srate=250,
                splitMethod=StratifiedKFold,
                kFold=10, 
                subs=9, 
                *a, 
                **kw)(*args, **kwargs)
    ```

    Note
    ----
    More details to see the codes.
    '''
    def __init__(self,
                 built_in,
                 dataGent,
                 splitMethod=StratifiedKFold,
                 beg=0,
                 end=4,
                 srate=250,
                 kFold=10,
                 shuffle=False,
                 random_state=None,
                 subs=9,
                 isCropped=False,
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 *args,
                 **kwargs):
        self.built_in = built_in
        self.dataGent = dataGent(beg=beg,
                                 end=end,
                                 srate=srate,
                                 *args,
                                 **kwargs)
        self.beg = beg
        self.end = end
        self.srate = srate
        self.splitMethod = splitMethod
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.subs = subs + 1
        self.isCropped = isCropped
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self.modelstr = built_in.__name__[7:]

    def __call__(self, *args, **kwargs):
        if self.isCropped:
            gent = self._read_cropped_data
            pass
        else:
            gent = self._read_data
            pass

        if not os.path.exists('model'):
            os.makedirs('model')

        model = self.built_in(*args, **kwargs, Samples=self.Samples)
        # save initial weights
        model.save_weights(self.modelstr + '.h5')

        tm = time.localtime()

        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=self.patience,
                                      verbose=0,
                                      mode='auto')

        avg_acc = []
        for i in range(1, self.subs):
            accik = []
            k = 0  # count kFolds
            for data in gent(i):
                k = k + 1

                filepath = os.path.join(
                    'model',
                    str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' +
                    str(tm.tm_mday) + '_' + str(tm.tm_hour) + '_' +
                    str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' + str(i) +
                    'T_' + self.modelstr + '(' + str(k) + ').h5')
                checkpointer = ModelCheckpoint(filepath=filepath,
                                               verbose=1,
                                               save_best_only=True)

                validation_name = 'Cross Validation'
                if data['x_val'] is None:
                    data['x_val'] = data['x_test']
                    data['y_val'] = data['y_test']
                    validation_name = 'Average Validation'

                history = model.fit(
                    x=data['x_train'],
                    y=data['y_train'],
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=[checkpointer, earlystopping],
                    verbose=self.verbose,
                    validation_data=[data['x_val'], data['y_val']]).history

                model.load_weights(filepath)
                loss, acc = model.evaluate(data['x_test'],
                                           data['y_test'],
                                           batch_size=self.batch_size,
                                           verbose=self.verbose)

                filepath = os.path.join(
                    'model',
                    str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' +
                    str(tm.tm_mday) + '_' + str(tm.tm_hour) + '_' +
                    str(tm.tm_min) + '_' + str(tm.tm_sec) + '_A0' + str(i) +
                    'T_' + self.modelstr + '(' + str(k) + ').npy')
                np.save(filepath, history)

                # reset layers weights to train a new one next fold.
                # you can't use model.reset_states() because it has
                # loaded another weights before.
                model.load_weights(self.modelstr + '.h5')

                accik.append(acc)
            avg_acc.append(np.average(np.asarray(accik)))
            del data
        total_avg_acc = np.average(np.asarray(avg_acc))
        print('{:d}-fold ' + validation_name + ' Accuracy'.format(self.kFold))
        for i in range(1, self.subs):
            print('Subject {0:0>2d}: {1:.2%}'.format(i, avg_acc[i - 1]))
        print('Average   : {:.2%}'.format(total_avg_acc))
        avg_acc.append(total_avg_acc)
        return avg_acc

    def getConfig(self):
        print(
            'Method: {0:s}\nSplit Method: {1:s}\nCross Validation Fold: {2:d}\n'
            'shuffle: {3}\nrandom_state: {4:d}\nNumber of subjects: {5:d}'.
            format(self.built_in.__name__, self.splitMethod.__name__,
                   self.kFold, self.shuffle, self.random_state, self.subs))

    def setConfig(self,
                  built_in,
                  dataGent,
                  splitMethod=StratifiedKFold,
                  beg=0,
                  end=4,
                  srate=250,
                  kFold=10,
                  shuffle=False,
                  random_state=None,
                  subs=9,
                  isCropped=False,
                  batch_size=10,
                  epochs=300,
                  patience=100,
                  verbose=2,
                  *args,
                  **kwargs):
        self.built_in = built_in
        self.dataGent = dataGent(*args,
                                 **kwargs,
                                 beg=beg,
                                 end=end,
                                 srate=srate)
        self.beg = beg
        self.end = end
        self.srate = srate
        self.splitMethod = splitMethod
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.subs = subs + 1
        self.isCropped = isCropped
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self.modelstr = built_in.__name__[7:]

    def _read_data(self, subject):
        '''
        Read data from dataGent.

        Parameters
        ----------
        ```txt
        subject : int, Identifier of subject.
        ```

        Yields
        ------
        ```txt
        data    : dict, Includes train, val and test data.
        ```
        '''
        data = {
            'x_train': None,
            'x_val': None,
            'x_test': None,
            'y_train': None,
            'y_val': None,
            'y_test': None
        }
        filepath = os.path.join('data', '4s', 'Test',
                                'A0' + str(subject) + 'E.mat')
        data['x_test'] = self.dataGent(filepath, label=False)
        data['y_test'] = self.dataGent(filepath, label=True)
        filepath = os.path.join('data', '4s', 'Train',
                                'A0' + str(subject) + 'T.mat')
        for (data['x_train'],
             data['y_train']), (data['x_val'], data['y_val']) in self._spilt(
                 self.dataGent(filepath, label=False),
                 self.dataGent(filepath, label=True)):
            yield data

    def _read_cropped_data(self, subject):
        '''
        Read cropped data from dataGent.

        TODO: Should be completed, can't work now.

        Parameters
        ----------
        ```txt
        subject : int, Identifier of subject.
        ```
        
        Yields
        ------
        ```txt
        data    : dict, Includes train, val and test data.
        ```
        '''
        data = {
            'x_train': None,
            'x_val': None,
            'x_test': None,
            'y_train': None,
            'y_val': None,
            'y_test': None
        }
        filepath = os.path.join('data', '4s', 'Test',
                                'A0' + str(subject) + 'E.mat')
        data['x_test'] = self.dataGent(filepath,
                                       label=False,
                                       *self.args,
                                       **self.kwargs)
        data['y_test'] = self.dataGent(filepath,
                                       label=True,
                                       *self.args,
                                       **self.kwargs)
        filepath = os.path.join('data', '4s', 'Train',
                                'A0' + str(subject) + 'T.mat')
        for (data['x_train'],
             data['y_train']), (data['x_val'], data['y_val']) in self._spilt(
                 self.dataGent(filepath,
                               label=False,
                               *self.args,
                               **self.kwargs),
                 self.dataGent(filepath, label=True, *self.args,
                               **self.kwargs)):
            yield data

    def _spilt(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Action depends on the split method you choose.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        val : ndarray
            The validating set indices for that split.
        """
        sm = self.splitMethod(n_splits=self.kFold,
                              shuffle=self.shuffle,
                              random_state=self.random_state)
        for train_index, val_index in sm.split(X, y, groups):
            # (x_train, y_train), (x_val, y_val)
            if not train_index.any():
                raise ValueError('Training data shouldn\'t be empty.')
            elif not val_index.any():
                yield (X[train_index], y[train_index]), (None, None)
            else:
                yield (X[train_index], y[train_index]), (X[val_index],
                                                         y[val_index])
