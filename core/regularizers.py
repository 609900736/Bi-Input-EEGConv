# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.regularizers import Regularizer
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
tf.keras.layers.ActivityRegularization


class TSG(Regularizer):
    '''
    Regularizer for L1, L2, L12, L21 and TL1 regularization.

    Parameters
    ----------
    ```txt
    l1              : float, Positive L1 regularization factor.
    l2              : float, Positive L2 regularization factor.
    l12             : float, Positive L12 regularization factor.
    l21             : float, Positive L21 regularization factor.
    tl1             : float, Positive TL1 regularization factor.
    ```

    Return
    ------
    ```txt
    regularization  : float, Regularization fine.
    ```
    '''
    def __init__(self, l1=0., l2=0., l12=0., l21=0., tl1=0.):  # pylint: disable=redefined-outer-name
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l12 = K.cast_to_floatx(l12)
        self.l21 = K.cast_to_floatx(l21)
        self.tl1 = K.cast_to_floatx(tl1)

    def __call__(self, x):
        if not self.l1 and not self.l2 and not self.l12 and not self.l21 and not self.tl1:
            return K.constant(0.)
        regularization = 0.
        if self.l1:
            regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        if self.l2:
            regularization += self.l2 * math_ops.reduce_sum(math_ops.square(x))
        if self.l12:
            regularization += self.l12 * math_ops.sqrt(
                math_ops.reduce_sum(
                    math_ops.square(math_ops.reduce_sum(math_ops.abs(x), 1))))
        if self.l21:
            regularization += self.l21 * math_ops.reduce_sum(
                math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x), 1)))
        if self.tl1:# for feature selection matrix shapes (None, features, timesteps)
            regularization += self.tl1 * math_ops.reduce_sum(
                math_ops.abs(math_ops.sub(x[:, :, :-1], x[:, :, 1:])))
        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'l2': float(self.l2),
            'l12': float(self.l12),
            'l21': float(self.l21),
            'tl1': float(self.tl1)
        }


class TSGRegularization(Layer):
    """
    Layer that applies an update to the cost function based input activity.

    Arguments:
        l1  : float, Positive L1 regularization factor.
        l2  : float, Positive L2 regularization factor.
        l12 : float, Positive L12 regularization factor.
        l21 : float, Positive L21 regularization factor.
        tl1 : float, Positive TL1 regularization factor.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
  """
    def __init__(self, l1=0., l2=0., l12=0., l21=0., tl1=0., **kwargs):
        super().__init__(activity_regularizer=TSG(l1=l1,
                                                  l2=l2,
                                                  l12=l12,
                                                  l21=l21,
                                                  tl1=tl1),
                         **kwargs)
        self.supports_masking = True
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l12 = K.cast_to_floatx(l12)
        self.l21 = K.cast_to_floatx(l21)
        self.tl1 = K.cast_to_floatx(tl1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'l1': self.l1, 'l2': self.l2}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def l1(l1=0.01):
    return TSG(l1=l1)


def l2(l2=0.01):
    return TSG(l2=l2)


def l1_l2(l1=0.01, l2=0.01):
    return TSG(l1=l1, l2=l2)


def l2_1(l2_1=0.01):
    return TSG(l2_1=l2_1)


def tl1(tl1=0.01):
    return TSG(tl1=tl1)


def sgl(l1=0.01, l21=0.01):
    return TSG(l1=l1, l21=l21)


def tsg(l1=0.01, l21=0.01, tl1=0.01):
    return TSG(l1=l1, l21=l21, tl1=tl1)