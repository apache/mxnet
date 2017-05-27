#!/usr/bin/env python
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning."""
from __future__ import absolute_import

from .context import Context, current_context, cpu, gpu
from .base import MXNetError
from . import base
from . import contrib
from . import ndarray
from . import name
# use mx.sym as short for symbol
from . import symbol as sym
from . import symbol
from . import symbol_doc
from . import io
from . import recordio
from . import operator
# use mx.nd as short for mx.ndarray
from . import ndarray as nd
# use mx.rnd as short for mx.random
from . import random as rnd
from . import random
from . import optimizer
from . import model
from . import notebook
from . import initializer
# use mx.init as short for mx.initializer
from . import initializer as init
from . import visualization
# use viz as short for mx.ndarray
from . import visualization as viz
from . import callback
# from . import misc
from . import lr_scheduler
# use mx.kv as short for kvstore
from . import kvstore as kv
from . import kvstore_server
# Runtime compile module
from .rtc import Rtc as rtc
# Attribute scope to add attributes to symbolic graphs
from .attribute import AttrScope

from . import monitor
from . import monitor as mon

from . import torch
from . import torch as th

from . import profiler
from . import log

from . import module
from . import module as mod

from . import image
from . import image as img

from . import test_utils

from . import rnn

__version__ = base.__version__
