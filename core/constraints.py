# coding:utf-8
"""
TODO: kernel constraint to keep mean of kernel weights values 0, 
      and variance of them values 1/K (where K means the number 
      of input's neurons).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_core.python.keras.constraints import Constraint
from tensorflow_core.python.keras import backend as K


class StdNorm(Constraint):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, w):
        k = tf.shape(w)[-2]
        mu = tf.reduce_mean(w, axis=self.axis, keepdims=True)
        std = tf.reduce_std(w, axis=self.axis, keepdims=True)
        mu = tf.multiply(tf.ones_like(w), mu)
        std = tf.multiply(tf.ones_like(w), std)
        return tf.divide(tf.multiply(tf.subtract(w, mu), tf.sqrt(k)), std)

    def get_config(self):
        return {'axis': self.axis}


std_norm = StdNorm