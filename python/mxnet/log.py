#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Logging utilities."""
import logging
import sys

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
        else:
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
        else:
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
    """Get customized logger.

    Args:
        name: Name of the logger.
        level: Level to log.

    Returns:
        A logger.
    """
    logger = logging.getLogger(name)
    if name is not None and not getattr(logger, '_init_done', None):
        logger._init_done = True
        if filename:
            mode = filemode if filemode else 'a'
            hdlr = logging.FileHandler(filename, mode)
        else:
            hdlr = logging.StreamHandler()
            # the `_Formatter` contain some escape character to
            # represent color, which is not suitable for FileHandler,
            # (TODO) maybe we can add another Formatter for FileHandler.
            hdlr.setFormatter(_Formatter())
        logger.addHandler(hdlr)
        logger.setLevel(level)
    return logger
