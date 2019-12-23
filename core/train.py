# coding:utf-8

import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.utils import load_data, load_or_gen_filterbank_data, load_locs, load_or_gen_interestingband_data, load_or_generate_images, highpassfilter, bandpassfilter
from core.models import EEGNet, rawEEGConvNet, graphEEGConvNet, BiInputsEEGConvNet, ShallowConvNet, DeepConvNet, MB3DCNN
from core.splits import StratifiedKFold
from core.callbacks import MyModelCheckpoint, EarlyStopping

console = sys.stdout


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
                  F=9,
                  D=4,
                  Ns=4,
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']):
    model = EEGNet(nClasses,
                   Chans=Chans,
                   Samples=Samples,
                   F1=F,
                   D=D,
                   F2=nClasses * 2 * Ns)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    # export graph of the model
    tf.keras.utils.plot_model(model, 'EEGNet.png', show_shapes=True)
    return model


def create_rawEEGConvNet(nClasses,
                         Samples,
                         Chans=22,
                         Colors=1,
                         F=9,
                         D=4,
                         Ns=4,
                         l1=0.001,
                         l21=0.001,
                         tl1=0.001,
                         optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy']):
    model = rawEEGConvNet(nClasses,
                          Chans=Chans,
                          Samples=Samples,
                          Colors=Colors,
                          F1=F,
                          D=D,
                          F2=nClasses * 2 * Ns,
                          l1=l1,
                          l21=l21,
                          tl1=tl1)
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


def create_biEEGConvNet():
    """
    TODO: 
    """
    pass


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
    built_fn        : function, Create Training model which need to cross validate.
                      Please use `create_` at the begining of function name, 
                      like `create_modelname`.
    dataGent        : class, Generate data for @built_fn, shapes (n_trails, ...). 
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
    cropping        : bool, Switch of cropped training. Default = False.
    normalizing     : bool, Switch of normalizing data. Default = True.
    batch_size      : int, Batch size.
    epochs          : int, Training epochs.
    patience        : int, Early stopping patience.
    verbose         : int, One of 0, 1 and 2.
    *a, *args       : tuple, Parameters used by @dataGent and @built_fn respectively
    **kw, **kwargs  : dict, Parameters used by @dataGent and @built_fn respectively, 
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
        return keras_model

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
                create_model, 
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
                 built_fn,
                 dataGent,
                 splitMethod=StratifiedKFold,
                 beg=0,
                 end=4,
                 srate=250,
                 kFold=10,
                 shuffle=False,
                 random_state=None,
                 subs=9,
                 cropping=False,
                 normalizing=True,
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 *args,
                 **kwargs):
        self.built_fn = built_fn
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
        self.subs = subs
        self.cropping = cropping
        self.normalizing = normalizing
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self.modelstr = built_fn.__name__[7:]

    def __call__(self, *args, **kwargs):
        if self.cropping:
            gent = self._read_cropped_data
            pass
        else:
            gent = self._read_data
            pass

        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('result'):
            os.makedirs('result')
        validation_name = 'Cross Validation'

        model = self.built_fn(*args, **kwargs, Samples=self.Samples)
        # save initial weights
        model.save_weights(self.modelstr + '.h5')

        tm = time.localtime()

        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=self.patience,
                                      verbose=0,
                                      mode='auto')

        avg_acc = []
        for i in range(1, self.subs + 1):
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
                checkpointer = MyModelCheckpoint(filepath=filepath,
                                                 verbose=1,
                                                 save_best_only=True,
                                                 statistic_best=True,
                                                 p=0.05)

                if data['x_val'] is None:
                    data['x_val'] = data['x_test']
                    data['y_val'] = data['y_test']
                    if k == 1:
                        validation_name = 'Average Validation'

                if self.normalizing:
                    data = self._normalize(data)

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

                filepath = filepath[:-3] + '.npy'
                np.save(filepath, history)

                # reset layers weights to train a new one next fold.
                # you can't use model.reset_states() because it has
                # loaded another weights before.
                model.load_weights(self.modelstr + '.h5')

                accik.append(acc)
            avg_acc.append(np.average(np.asarray(accik)))
            del data
        total_avg_acc = np.average(np.asarray(avg_acc))
        filepath = os.path.join(
            'result',
            str(tm.tm_year) + '_' + str(tm.tm_mon) + '_' + str(tm.tm_mday) +
            '_' + str(tm.tm_hour) + '_' + str(tm.tm_min) + '_' +
            str(tm.tm_sec) + '_' + self.modelstr + '.txt')
        with open(filepath, 'w+') as f:
            sys.stdout = f
            print(('{0:s} {1:d}-fold ' + validation_name + ' Accuracy').format(
                self.modelstr, self.kFold))
            for i in range(1, self.subs + 1):
                print('Subject {0:0>2d}: {1:.2%}'.format(i, avg_acc[i - 1]))
            print('Average   : {:.2%}'.format(total_avg_acc))
            sys.stdout = console
            f.seek(0, 0)
            print(f.readlines())
            f.close()
        avg_acc.append(total_avg_acc)
        return avg_acc

    def getConfig(self):
        print(
            'Method: {0:s}\nSplit Method: {1:s}\nCross Validation Fold: {2:d}\n'
            'shuffle: {3}\nrandom_state: {4:d}\nNumber of subjects: {5:d}'.
            format(self.modelstr, self.splitMethod.__name__, self.kFold,
                   self.shuffle, self.random_state, self.subs))

    def setConfig(self,
                  built_fn,
                  dataGent,
                  splitMethod=StratifiedKFold,
                  beg=0,
                  end=4,
                  srate=250,
                  kFold=10,
                  shuffle=False,
                  random_state=None,
                  subs=9,
                  cropping=False,
                  normalizing=True,
                  batch_size=10,
                  epochs=300,
                  patience=100,
                  verbose=2,
                  *args,
                  **kwargs):
        self.built_fn = built_fn
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
        self.subs = subs
        self.cropping = cropping
        self.normalizing = normalizing
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self.modelstr = built_fn.__name__[7:]

    def _normalize(self, data: dict):
        '''Normalizing on each trial, supports np.nan numbers'''
        # TODO: need to reconsider the implement
        meta = ['x_train', 'x_test', 'x_val']
        for s in meta:
            if not s in data:
                raise ValueError('Wrong using crossValidate._normalize(data),'
                                 ' data is a dict which should have `x_train`'
                                 ', `x_test`, and `x_val` keys')

        for s in meta:
            temp = data[s]
            for k in range(temp.shape[0]):
                mu = np.nanmean(temp[k])
                std = np.nanstd(temp[k])
                temp[k] = (temp[k] - mu) / std
            data[s] = temp

        return data

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
