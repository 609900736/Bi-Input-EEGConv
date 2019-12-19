# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
from tensorflow_core.python.keras.regularizers import Regularizer
from tensorflow_core.python.keras.regularizers import l1 as l_1
from tensorflow_core.python.keras.regularizers import l2 as l_2
from tensorflow_core.python.keras.regularizers import l1_l2
from tensorflow_core.python.keras import backend as K


class TSG(Regularizer):
    '''
    Regularizer for TSG regularization.

    Parameters
    ----------
    ```txt
    l1              : float, Positive L1 regularization factor.
    l21             : float, Positive L21 regularization factor.
    tl1             : float, Positive TL1 regularization factor.
    ```

    Return
    ------
    ```txt
    regularization  : float, Regularization fine.
    ```
    '''
    def __init__(self, l1=0., l21=0., tl1=0.):  # pylint: disable=redefined-outer-name
        self.l1 = K.cast_to_floatx(l1)
        self.l21 = K.cast_to_floatx(l21)
        self.tl1 = K.cast_to_floatx(tl1)

    @tf.function
    def __call__(self, x):
        if not self.l1 and not self.l21 and not self.tl1:
            return K.constant(0.)
        regularization = 0.

        if array_ops.rank(x) == 4:  # shape (?, 1, Timesteps, Features)
            ntf = math_ops.reduce_sum(x, 1)  # shape (?, Timesteps, Features)
        elif array_ops.rank(x) == 5:  # shape (?, 1, 1, Inputs, Outputs)
            ntf = math_ops.reduce_sum(x, [1, 2])  # shape (?, Inputs, Outputs)
        else:
            ntf = x  # shape (?, Inputs, Outputs)

        for n in math_ops.range(array_ops.shape(ntf)[0]):
            if self.l1:
                regularization += self.l1 * math_ops.reduce_sum(
                    math_ops.abs(ntf[n, :, :]))
            if self.l21:
                regularization += self.l21 * math_ops.reduce_sum(
                    math_ops.multiply(
                        math_ops.sqrt(
                            math_ops.to_float(array_ops.shape(ntf)[1])),
                        math_ops.sqrt(
                            math_ops.reduce_sum(math_ops.square(ntf[n, :, :]),
                                                1))))
            if self.tl1:
                regularization += self.tl1 * math_ops.reduce_sum(
                    math_ops.abs(math_ops.subtract(ntf[n, :-1, :], ntf[n, 1:, :])))
        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'l21': float(self.l21),
            'tl1': float(self.tl1)
        }


# l2_1 = group lasso
def l2_1(l21=0.01):
    '''group lasso'''
    return TSG(l21=l21)


# to preserve the temporal smoothness
def tsc(tl1=0.01):
    '''to preserve the temporal smoothness'''
    return TSG(tl1=tl1)


# l1 + l2_1 = sparse group lasso
def sgl(l1=0.01, l21=0.01):
    '''sparse group lasso'''
    return TSG(l1=l1, l21=l21)


def tsg(l1=0.01, l21=0.01, tl1=0.01):
    '''temporal constrained sparse group lasso'''
    return TSG(l1=l1, l21=l21, tl1=tl1)