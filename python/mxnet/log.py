#!/usr/bin/env python

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

# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Logging utilities."""
import logging
import sys
import warnings

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

PY3 = sys.version_info[0] == 3


class _Formatter(logging.Formatter):
    # pylint: disable= no-self-use
    """Customized log formatter."""

    def __init__(self):
        datefmt = '%m%d %H:%M:%S'
        super(_Formatter, self).__init__(datefmt=datefmt)

    def _get_color(self, level):
        # pylint: disable= missing-docstring
        if logging.WARNING <= level:
            return '\x1b[31m'
        elif logging.INFO <= level:
            return '\x1b[32m'
        return '\x1b[34m'

    def _get_label(self, level):
        # pylint: disable= missing-docstring
        if level == logging.CRITICAL:
            return 'C'
        elif level == logging.ERROR:
            return 'E'
        elif level == logging.WARNING:
            return 'W'
        elif level == logging.INFO:
            return 'I'
        elif level == logging.DEBUG:
            return 'D'
        return 'U'

    def format(self, record):
        # pylint: disable= missing-docstring
        fmt = self._get_color(record.levelno)
        fmt += self._get_label(record.levelno)
        fmt += '%(asctime)s %(process)d %(pathname)s:%(funcName)s:%(lineno)d'
        fmt += ']\x1b[0m'
        fmt += ' %(message)s'
        if PY3:
            self._style._fmt = fmt # pylint: disable= no-member
        else:
            self._fmt = fmt
        return super(_Formatter, self).format(record)

def getLogger(name=None, filename=None, filemode=None, level=WARNING):
    """Gets a customized logger.

    .. note:: `getLogger` is deprecated. Use `get_logger` instead.

    """
    warnings.warn("getLogger is deprecated, Use get_logger instead.",
                  DeprecationWarning, stacklevel=2)
    return get_logger(name, filename, filemode, level)

def get_logger(name=None, filename=None, filemode=None, level=WARNING):
    """Gets a customized logger.

    Parameters
    ----------
    name: str, optional
        Name of the logger.
    filename: str, optional
        The filename to which the logger's output will be sent.
    filemode: str, optional
        The file mode to open the file (corresponding to `filename`),
        default is 'a' if `filename` is not ``None``.
    level: int, optional
        The `logging` level for the logger.
        See: https://docs.python.org/2/library/logging.html#logging-levels

    Returns
    -------
    Logger
        A customized `Logger` object.

    Example
    -------
    ## get_logger call with default parameters.
    >>> from mxnet.log import get_logger
    >>> logger = get_logger("Test")
    >>> logger.warn("Hello World")
    W0505 00:29:47 3525 <stdin>:<module>:1] Hello World

    ## get_logger call with WARNING level.
    >>> import logging
    >>> logger = get_logger("Test2", level=logging.WARNING)
    >>> logger.warn("Hello World")
    W0505 00:30:50 3525 <stdin>:<module>:1] Hello World
    >>> logger.debug("Hello World") # This doesn't return anything as the level is logging.WARNING.

    ## get_logger call with DEBUG level.
    >>> logger = get_logger("Test3", level=logging.DEBUG)
    >>> logger.debug("Hello World") # Logs the debug output as the level is logging.DEBUG.
    D0505 00:31:30 3525 <stdin>:<module>:1] Hello World
    """
    logger = logging.getLogger(name)
    if name is not None and not getattr(logger, '_init_done', None):
        logger._init_done = True
        if filename:
            mode = filemode if filemode else 'a'
            hdlr = logging.FileHandler(filename, mode)
        else:
            hdlr = logging.StreamHandler() # pylint: disable=redefined-variable-type
            # the `_Formatter` contain some escape character to
            # represent color, which is not suitable for FileHandler,
            # (TODO) maybe we can add another Formatter for FileHandler.
            hdlr.setFormatter(_Formatter())
        logger.addHandler(hdlr)
        logger.setLevel(level)
    return logger
