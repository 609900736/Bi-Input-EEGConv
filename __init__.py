# coding:utf-8
"""
Model BIEEGConv

@author: Boxann John
@date: 11/07/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core import models
from .core import train
from .core import utils
from .core import regularizers
from .core import generators
from .core import splits
from .core import layers
from .core import callbacks
from .core import visualization

import BCIIV2a as example


__name__ = 'BIEEGConv'
__all__ = {
    'models', 'train', 'utils', 'regularizers', 'generators', 'splits',
    'layers', 'callbacks', 'visualization'
}

__version__ = '0.0.1-rc'
