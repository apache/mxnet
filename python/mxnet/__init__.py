#!/usr/bin/env python
# pylint: disable=invalid-name, protected-access
# coding: utf-8
"""MXNet: a concise, fast and flexible framework for deep learning

MXNet is a project that evolves from cxxnet, minerva and purine2.
The interface is designed in collaboration by authors of three projects.

"""
from __future__ import absolute_import

from .context import Context, current_context
from .narray import NArray
from .function import _FunctionRegistry
from .symbol import Symbol
from .symbol_creator import _SymbolCreatorRegistry

__version__ = "0.1.0"

# this is a global function registry that can be used to invoke functions
op = NArray._init_function_registry(_FunctionRegistry())
sym = Symbol._init_symbol_creator_registry(_SymbolCreatorRegistry())
