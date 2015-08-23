#!/usr/bin/env python
# pylint: disable=invalid-name, protected-access
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning

MXNet is a project that evolves from cxxnet, minerva and purine2.
The interface is designed in collaboration by authors of three projects.

"""
from __future__ import absolute_import

from .context import Context, current_context
from . import narray
from . import symbol

__version__ = "0.1.0"



