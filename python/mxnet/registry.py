# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=no-member

"""Registry for serializable objects."""

import json
import warnings

from .base import string_types

_REGISTRY = {}


def get_registry(base_class):
    """Get a copy of the registry.

    Parameters
    ----------
    base_class : type
        base class for classes that will be registered.

    Returns
    -------
    a registrator
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    return _REGISTRY[base_class].copy()


def get_register_func(base_class, nickname):
    """Get registrator function.

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a registrator function
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    registry = _REGISTRY[base_class]

    def register(klass, name=None):
        """Register functions"""
        assert issubclass(klass, base_class), \
            "Can only register subclass of %s"%base_class.__name__
        if name is None:
            name = klass.__name__
        name = name.lower()
        if name in registry:
            warnings.warn(
                "\033[91mNew %s %s.%s registered with name %s is"
                "overriding existing %s %s.%s\033[0m"%(
                    nickname, klass.__module__, klass.__name__, name,
                    nickname, registry[name].__module__, registry[name].__name__),
                UserWarning, stacklevel=2)
        registry[name] = klass
        return klass

    register.__doc__ = "Register %s to the %s factory"%(nickname, nickname)
    return register


def get_alias_func(base_class, nickname):
    """Get registrator function that allow aliases.

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a registrator function
    """
    register = get_register_func(base_class, nickname)

    def alias(*aliases):
        """alias registrator"""
        def reg(klass):
            """registrator function"""
            for name in aliases:
                register(klass, name)
            return klass
        return reg
    return alias


def get_create_func(base_class, nickname):
    """Get creator function

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a creator function
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    registry = _REGISTRY[base_class]

    def create(*args, **kwargs):
        """Create instance from config"""
        if len(args):
            name = args[0]
            args = args[1:]
        else:
            name = kwargs.pop(nickname)

        if isinstance(name, base_class):
            assert len(args) == 0 and len(kwargs) == 0, \
                "%s is already an instance. Additional arguments are invalid"%(nickname)
            return name

        if isinstance(name, dict):
            return create(**name)

        assert isinstance(name, string_types), "%s must be of string type"%nickname

        if name.startswith('['):
            assert not args and not kwargs
            name, kwargs = json.loads(name)
            return create(name, **kwargs)
        elif name.startswith('{'):
            assert not args and not kwargs
            kwargs = json.loads(name)
            return create(**kwargs)

        name = name.lower()
        assert name in registry, \
            "%s is not registered. Please register with %s.register first"%(
                str(name), nickname)
        return registry[name](*args, **kwargs)

    create.__doc__ = """Create a %s instance from config.

Parameters
----------
%s : str or %s instance
    class name of desired instance. If is a instance,
    it will be returned directly.
**kwargs : dict
    arguments to be passed to constructor"""%(nickname, nickname, base_class.__name__)

    return create
