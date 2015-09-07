# coding: utf-8
""" KVStore in mxnet """

from __future__ import absolute_import

def init(contexts):
    """ Init key-value store with a list of context

    Parameters
    ----------
    contexts : list of Context
       The list of local devices used by this process
    """

def insert(key, value):
    """ Insert a key-value pair into the store

    Parameters
    ----------
    key : int
        The key
    value : NArray
        The value
    """

def push(key, value):
    """ Push a value into the store

    Parameters
    ----------
    key : int
        The key
    value : NArray
        The value
    """

def pull(key, value):
    """ Pull the value from the store

    Parameters
    ----------
    key : int
        The key
    value : NArray
        The value
    """

def register(updater):
    """ Register a updater into the store

    Example:
    def Update(grad, weight):
        weight[:] -= lr * grad  / batch_size

    Parameters
    ----------

    """
