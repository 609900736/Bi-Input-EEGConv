#coding:utf-8

import os
import numpy as np
import math as m
import tensorflow as tf
import tensorflow.keras as keras

import core.utils as utils

#keras.layers
class BiInputConv(tf.keras.Model):
    """
    """
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)
    
def BiInputEEGConv(nb_classes, Chans = 64, Samples = 128,
                   dropoutRate = 0.5, kernLength = 64, F1 = 8,
                   D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """
    Inputs:
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
      """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    s = tf.keras.Input()
    g = tf.keras.Input()
    s = tf.keras.layers.Conv2D(F1, (1, kernLength), padding = 'same',
                               input_shape = (Chans, Samples, 1),
                               use_bias = False)(s)
    g = tf.keras.layers.SeparableConv2D(F1, (3, 3), padding = 'same',
                               input_shape = (Chans, Samples, 1),
                               use_bias = False)(g)
    model = tf.keras.Model()
    model.fit()


if __name__=='__main__':
    print(tf.keras.__version__)