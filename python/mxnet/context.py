# coding: utf-8
""" code for context management """
from __future__ import absolute_import

class Context(object):
    """Context representing device and device id in mxnet"""
    # static class variable
    default_ctx = None
    devmask2type = {1: 'cpu', 2: 'gpu'}
    devtype2mask = {'cpu': 1, 'gpu': 2}

    def __init__(self, device_type, device_id=0):
        """Constructing a context.

        Parameters
        ----------
        device_type : str (can be 'cpu' or 'gpu')
            a string representing the device type

        device_id : int (default=0)
            the device id of the device, needed for GPU
        """
        self.device_mask = Context.devtype2mask[device_type]
        self.device_id = device_id
        self._old_ctx = None

    @property
    def device_type(self):
        """Return device type of current context.

        Returns
        -------
        device_type : str
        """
        return Context.devmask2type[self.device_mask]

    def __str__(self):
        return 'Context(device_type=%s, device_id=%d)' % (
            self.device_type, self.device_id)

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        self._old_ctx = Context.default_ctx
        Context.default_ctx = self
        return self

    def __exit__(self, ptype, value, trace):
        Context.default_ctx = self._old_ctx

# initialize the default context in Context
Context.default_ctx = Context('cpu', 0)


def cpu(device_id=0):
    """Return a CPU context.

    This function is a short cut for Context('cpu', device_id)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. device_id is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    context : Context
        The corresponding CPU context.
    """
    return Context('cpu', device_id)


def gpu(device_id=0):
    """Return a GPU context.

    This function is a short cut for Context('cpu', device_id)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device, needed for GPU

    Returns
    -------
    context : Context
        The corresponding GPU context.
    """
    return Context('gpu', device_id)


def current_context():
    """Return the current context.

    Returns
    -------
    default_ctx : Context
    """
    return Context.default_ctx
