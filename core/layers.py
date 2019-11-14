# coding:utf-8
from tensorflow.python.keras.layers import Layer, Attention

from core.regularizers import TSG


# TODO: Construct attention layers
class BaseAttention(Layer):
    def __init__(self,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rawEEGAttention(BaseAttention):
    def __init__(self,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, mask=None):
        return super().call(inputs, mask)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class graphEEGAttention(BaseAttention):
    def __init__(self,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable=trainable,
                         name=name,
                         dtype=dtype,
                         dynamic=dynamic,
                         **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

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
    def __init__(self, l1=0., l2=0., l21=0., tl1=0., **kwargs):
        super().__init__(activity_regularizer=TSG(l1=l1,
                                                  l2=l2,
                                                  l21=l21,
                                                  tl1=tl1),
                         **kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2
        self.l21 = l21
        self.tl1 = tl1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'l1': self.l1,
            'l2': self.l2,
            'l21': self.l21,
            'tl1': self.tl1
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))