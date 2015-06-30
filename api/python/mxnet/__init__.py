#!/usr/bin/env python
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning

MXNet is a project that evolves from cxxnet, minerva and purine2.
The interface is designed in collaboration by authors of three projects.

Version : 0.10
"""
from __future__ import absolute_import

from .context import Context, current_context
from .narray import NArray, _init_function_registry
from .function import _FunctionRegistry

# this is a global function registry that can be used to invoke functions
op = _init_function_registry(_FunctionRegistry())
