# coding:utf-8
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
prep = False
Samples = (end - start) * srate
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    if prep:
        pp = '_pp'
    else:
        pp = ''

    for i in range(1, 10):
        filepath = os.path.join('data',
                                str(end) + 's', 'Test',
                                'A0' + str(i) + 'E' + pp + '.mat')
        x_test = load_data(filepath, label=False)
        x_test = np.expand_dims(
            x_test[:, :, int(start * srate):int(end * srate)], -1)
        filepath = os.path.join('data',
                                str(end) + 's', 'Test',
                                'A0' + str(i) + 'E_label' + pp + '.mat')
        y_test = load_data(filepath, label=True)

        filepath = os.path.join(
            'model',
            '2019_10_14_20_47_53' + '_A0' + str(i) + 'T_rawEEGConvNet.h5')
        model = load_model(filepath)
        #model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        #          loss=tf.keras.losses.sparse_categorical_crossentropy,
        #          metrics=['accuracy'])
        model.evaluate(x_test, y_test, batch_size=10, verbose=2)

    pass