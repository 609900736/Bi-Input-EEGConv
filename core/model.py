#coding:utf-8

import os
import numpy as np
import math as m
import tensorflow as tf
import tensorflow.python.keras.api._v1.keras as keras

import core.utils as utils

#keras.layers
class BiInputConv(tf.keras.Model):
    """
    """
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)
    
def BIEEGConv():
    tf.keras.layers

if __name__=='__main__':
    print(tf.keras.__version__)