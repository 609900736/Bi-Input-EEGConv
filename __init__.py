# coding:utf-8
"""
Model BIEEGConv

@author: Boxann John
@date: 2019/11/07
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core import models
from .core import train
from .core import utils
from .core import regularizers
from .core import generators

__name__ = 'BIEEGConv'
__all__ = {'models', 'train', 'utils', 'regularizers', 'generators'}

__version__ = '0.0.1-rc'
