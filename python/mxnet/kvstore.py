# coding: utf-8
""" KVStore in mxnet """

from __future__ import absolute_import

def init_devices(contexts):
    """ Init key-value store with a list of device contexts

    Parameters
    ----------
    contexts : list of Context
       The list of local devices used by this process
    """

def init(kv_list):
    """ Initialize a list of key-value pairs

    Parameters
    ----------
    kv_list : tuple or list/generator of tuples
        a key-value tuple or a list of key-value tuples
    """

def push(kv_list):
    """ Push a value into the store

    Parameters
    ----------
    """

def pull(kv_list):
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
