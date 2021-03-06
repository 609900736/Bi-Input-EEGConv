# coding:utf-8
import tensorflow as tf
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.dense_attention import Attention
from tensorflow.python.keras.layers.merge import Multiply
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras import backend as K

from core.regularizers import TSG as _TSG


# TODO: Construct attention layers
class BaseAttention(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        softmax = K.softmax(inputs, self.axis)
        return tf.multiply(inputs, softmax)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'axis':self.axis, 'supports_masking':self.supports_masking}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rawEEGAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class graphEEGAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseSelfAttention(BaseAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))\

    def _relationFunc(self):
        pass


class rawEEGSelfAttention(BaseSelfAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class graphEEGSelfAttention(BaseSelfAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TSGRegularization(Layer):
    """
    Layer that applies an update to the cost function based input activity.

    Arguments:
        l1  : float, Positive L1 regularization factor.
        l21 : float, Positive L21 regularization factor.
        tl1 : float, Positive TL1 regularization factor.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
  """
    def __init__(self, l1=0., l21=0., tl1=0., **kwargs):
        super().__init__(activity_regularizer=_TSG(l1=l1, l21=l21, tl1=tl1),
                         **kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l21 = l21
        self.tl1 = tl1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'l1': self.l1, 'l21': self.l21, 'tl1': self.tl1}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))