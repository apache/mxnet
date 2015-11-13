# coding: utf-8
"""Context management API of mxnet."""
from __future__ import absolute_import


class Context(object):
    """Constructing a context.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'} or Context.
        String representing the device type

    device_id : int (default=0)
        The device id of the device, needed for GPU

    work_load : int (default=1)
        The work_load in a list of devices. For example,
            if we have a list of device with work_load [1, 2, 1],
            the first and third device will take 1/4 work load,
            while the second device will take 1/2 work_load.

    Note
    ----
    Context can also be used a way to change default context.

    Examples
    --------
    >>> # array on cpu
    >>> cpu_array = mx.md.ones((2, 3))
    >>> # switch default context to GPU(2)
    >>> with mx.Context(mx.gpu(2)):
    >>>     gpu_array = mx.md.ones((2, 3))
    >>> gpu_array.context
    gpu(2)
    """
    # static class variable
    default_ctx = None
    devtype2str = {1: 'cpu', 2: 'gpu', 3: 'cpu_pinned'}
    devstr2type = {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3}

    def __init__(self, device_type, device_id=0, work_load=1):
        if isinstance(device_type, Context):
            self.device_typeid = device_type.device_typeid
            self.device_id = device_type.device_id
            self.work_load = device_type.work_load
        else:
            self.device_typeid = Context.devstr2type[device_type]
            self.device_id = device_id
            self.work_load = work_load
        self._old_ctx = None

    @property
    def device_type(self):
        """Return device type of current context.

        Returns
        -------
        device_type : str
        """
        return Context.devtype2str[self.device_typeid]

    def __str__(self):
        return '%s(%d)' % (self.device_type, self.device_id)

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


def cpu(device_id=0, work_load=1):
    """Return a CPU context.

    This function is a short cut for Context('cpu', device_id)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. device_id is not needed for CPU.
        This is included to make interface compatible with GPU.

    work_load : int (default=1)
        The work_load in a list of devices.

    Returns
    -------
    context : Context
        The corresponding CPU context.
    """
    return Context('cpu', device_id, work_load)


def gpu(device_id=0, work_load=1):
    """Return a GPU context.

    This function is a short cut for Context('cpu', device_id)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device, needed for GPU

    work_load : int (default=1)
        The work_load in a list of devices.

    Returns
    -------
    context : Context
        The corresponding GPU context.
    """
    return Context('gpu', device_id, work_load)


def current_context():
    """Return the current context.

    Returns
    -------
    default_ctx : Context
    """
    return Context.default_ctx
