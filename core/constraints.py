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
try:
    from tensorflow.python.ops import math_ops, array_ops
except:
    from tensorflow_core.python.ops import math_ops, array_ops
from tensorflow_core.python.keras.constraints import Constraint
from tensorflow_core.python.keras import backend as K


class StdNorm(Constraint):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, w):
        mu = math_ops.reduce_mean(w, axis=self.axis, keepdims=True)
        std = math_ops.reduce_std(w, axis=self.axis, keepdims=True)
        mu = math_ops.multiply(array_ops.ones_like(w), mu)
        std = math_ops.multiply(array_ops.ones_like(w), std)
        return math_ops.divide(math_ops.subtract(w, mu), std)

    def get_config(self):
        return {'axis': self.axis}


std_norm = StdNorm