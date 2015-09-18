#!/usr/bin/env python
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning

MXNet is a project that evolves from cxxnet, minerva and purine2.
The interface is designed in collaboration by authors of three projects.

"""
from __future__ import absolute_import

from .context import Context, current_context, cpu, gpu
from .base import MXNetError
from . import base
from . import ndarray
from . import symbol
from . import kvstore as kv
from . import io
# use mx.nd as short for mx.ndarray
from . import ndarray as nd
from . import random
from . import optimizer
from . import model
from . import initializer
import atexit

__version__ = "0.1.0"

def finalize():
    """Stop all the components in mxnet.

    There is no need to call this function.
    This function will be automatically called at module exit.
    """
    # pylint: disable=protected-access
    base.check_call(base._LIB.MXFinalize())
    kv._cleanup()

atexit.register(finalize)
