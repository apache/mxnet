#!/usr/bin/env python
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning

MXNet is a project that evolves from cxxnet, minerva and purine2.
The interface is designed in collaboration by authors of three projects.

"""
from __future__ import absolute_import

from .context import Context, current_context
from .base import MXNetError
from . import narray
from . import symbol
from . import kvstore
from . import io

__version__ = "0.1.0"
