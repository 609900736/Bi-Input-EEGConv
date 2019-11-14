# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.keras.regularizers import Regularizer
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class TSG(Regularizer):
    '''
    Regularizer for L1, L2, L21 and TL1 regularization.

    Parameters
    ----------
    ```txt
    l1              : float, Positive L1 regularization factor.
    l2              : float, Positive L2 regularization factor.
    l21             : float, Positive L21 regularization factor.
    tl1             : float, Positive TL1 regularization factor.
    ```

    Return
    ------
    ```txt
    regularization  : float, Regularization fine.
    ```
    '''
    def __init__(self, l1=0., l2=0., l21=0., tl1=0.):  # pylint: disable=redefined-outer-name
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l21 = K.cast_to_floatx(l21)
        self.tl1 = K.cast_to_floatx(tl1)

    def __call__(self, x):
        if not self.l1 and not self.l2 and not self.l21 and not self.tl1:
            return K.constant(0.)
        regularization = 0.
        if self.l1:
            regularization += self.l1 * math_ops.reduce_sum(math_ops.abs(x))
        if self.l2:
            regularization += self.l2 * math_ops.sqrt(
                math_ops.reduce_sum(math_ops.square(x)))
        if self.l21:  # shape (None, kH, kW, In, Out)
            regularization += self.l21 * math_ops.reduce_sum(
                math_ops.mul(
                    math_ops.sqrt(
                        math_ops.cast(array_ops.shape(x)[3],
                                      dtype=K.floatx())),
                    math_ops.sqrt(math_ops.reduce_sum(math_ops.square(x), 3))))
        if self.tl1:  # shape (None, 1, Timesteps, Features)
            regularization += self.tl1 * math_ops.reduce_sum(
                math_ops.abs(math_ops.sub(x[:, :, :-1, :], x[:, :, 1:, :])))
        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'l2': float(self.l2),
            'l21': float(self.l21),
            'tl1': float(self.tl1)
        }


# l1 = lasso
def l1(l1=0.01):
    '''lasso'''
    return TSG(l1=l1)


# l2 = ridge
def l2(l2=0.01):
    '''ridge'''
    return TSG(l2=l2)


# l1_l2 = elastic net
def l1_l2(l1=0.01, l2=0.01):
    '''elastic net'''
    return TSG(l1=l1, l2=l2)


# l2_1 = group lasso, for pointwise conv kernal regularization
def l2_1(l21=0.01):
    '''group lasso, for pointwise conv kernal regularization'''
    return TSG(l21=l21)


# to preserve the temporal smoothness, for seperable conv activity regularization
def tsc(tl1=0.01):
    '''to preserve the temporal smoothness, for seperable conv activity regularization'''
    return TSG(tl1=tl1)


# l1 + l21 = sparse group lasso, for pointwise conv kernal regularization
def sgl(l1=0.01, l21=0.01):
    '''sparse group lasso, for pointwise conv kernal regularization'''
    return TSG(l1=l1, l21=l21)


def tsg(l1=0.01, l21=0.01, tl1=0.01):
    '''Please use it carefully'''
    return TSG(l1=l1, l21=l21, tl1=tl1)