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
"""Device management API of mxnet."""
import contextvars
import ctypes
from .base import _LIB
from .base import check_call


class Device:
    """Constructs a device structure.

    MXNet can run operations on CPU and different GPUs.
    A Device class describes the device type and ID on which computation should be carried on.

    One can use mx.cpu and mx.gpu for short.

    See also
    ----------
    `How to run MXNet on multiple CPU/GPUs <http://mxnet.incubator.apache.org/api/faq/distributed_training>`
    for more details.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'} or Device.
        String representing the device type.

    device_id : int (default=0)
        The device id of the device, needed for GPU.

    Note
    ----
    Device can also be used as a way to change the default device.

    Examples
    --------
    >>> # array on cpu
    >>> cpu_array = mx.np.ones((2, 3))
    >>> # switch default Device to GPU(2)
    >>> with mx.Device(mx.gpu(2)):
    ...     gpu_array = mx.np.ones((2, 3))
    >>> gpu_array.device
    gpu(2)

    One can also explicitly specify the device when creating an array.

    >>> gpu_array = mx.np.ones((2, 3), mx.gpu(1))
    >>> gpu_array.device
    gpu(1)
    """
    devtype2str = {1: 'cpu', 2: 'gpu', 3: 'cpu_pinned', 5: 'cpu_shared'}
    devstr2type = {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3, 'cpu_shared': 5}
    def __init__(self, device_type, device_id=0):
        if isinstance(device_type, Device):
            self.device_typeid = device_type.device_typeid
            self.device_id = device_type.device_id
        else:
            self.device_typeid = Device.devstr2type[device_type]
            self.device_id = device_id
        self._old_ctx = None

    @property
    def device_type(self):
        """Returns the device type of current device.

        Examples
        -------
        >>> mx.device.current_device().device_type
        'cpu'
        >>> mx.current_device().device_type
        'cpu'

        Returns
        -------
        device_type : str
        """
        return Device.devtype2str[self.device_typeid]

    def __hash__(self):
        """Compute hash value of device for dictionary lookup"""
        return hash((self.device_typeid, self.device_id))

    def __eq__(self, other):
        """Compares two devices. Two devices are equal if they
        have the same device type and device id.
        """
        return isinstance(other, Device) and \
            self.device_typeid == other.device_typeid and \
            self.device_id == other.device_id

    def __str__(self):
        return f'{self.device_type}({self.device_id})'

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        # Token can't be pickled and Token.old_value is Token.MISSING if _current.get() uses default value
        self._old_ctx = _current.get()
        _current.set(self)
        return self

    def __exit__(self, ptype, value, trace):
        _current.set(self._old_ctx)

    def empty_cache(self):
        """Empties the memory cache for the current device.

        MXNet utilizes a memory pool to avoid excessive allocations.
        Calling empty_cache will empty the memory pool of the
        device. This will only free the memory of the unreferenced data.

        Examples
        -------
        >>> ctx = mx.gpu(0)
        >>> arr = mx.np.ones((200,200), ctx=ctx)
        >>> del arr
        >>> ctx.empty_cache() # forces release of memory allocated for arr
        """
        dev_type = ctypes.c_int(self.device_typeid)
        dev_id = ctypes.c_int(self.device_id)
        check_call(_LIB.MXStorageEmptyCache(dev_type, dev_id))


def cpu(device_id=0):
    """Returns a CPU device.

    This function is a short cut for ``Device('cpu', device_id)``.
    For most operations, when no device is specified, the default device is `cpu()`.

    Examples
    ----------
    >>> with mx.cpu():
    ...     cpu_array = mx.np.ones((2, 3))
    >>> cpu_array.device
    cpu(0)
    >>> cpu_array = mx.np.ones((2, 3), ctx=mx.cpu())
    >>> cpu_array.device
    cpu(0)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. `device_id` is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    device : Device
        The corresponding CPU device.
    """
    return Device('cpu', device_id)


def cpu_pinned(device_id=0):
    """Returns a CPU pinned memory device. Copying from CPU pinned memory to GPU
    is faster than from normal CPU memory.

    This function is a short cut for ``Device('cpu_pinned', device_id)``.

    Examples
    ----------
    >>> with mx.cpu_pinned():
    ...     cpu_array = mx.np.ones((2, 3))
    >>> cpu_array.device
    cpu_pinned(0)
    >>> cpu_array = mx.np.ones((2, 3), ctx=mx.cpu_pinned())
    >>> cpu_array.device
    cpu_pinned(0)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device. `device_id` is not needed for CPU.
        This is included to make interface compatible with GPU.

    Returns
    -------
    device : Device
        The corresponding CPU pinned memory device.
    """
    return Device('cpu_pinned', device_id)


def gpu(device_id=0):
    """Returns a GPU device.

    This function is a short cut for Device('gpu', device_id).
    The K GPUs on a node are typically numbered as 0,...,K-1.

    Examples
    ----------
    >>> cpu_array = mx.np.ones((2, 3))
    >>> cpu_array.device
    cpu(0)
    >>> with mx.gpu(1):
    ...     gpu_array = mx.np.ones((2, 3))
    >>> gpu_array.device
    gpu(1)
    >>> gpu_array = mx.np.ones((2, 3), ctx=mx.gpu(1))
    >>> gpu_array.device
    gpu(1)

    Parameters
    ----------
    device_id : int, optional
        The device id of the device, needed for GPU.

    Returns
    -------
    device : Device
        The corresponding GPU device.
    """
    return Device('gpu', device_id)


def num_gpus():
    """Query CUDA for the number of GPUs present.

    Raises
    ------
    Will raise an exception on any CUDA error.

    Returns
    -------
    count : int
        The number of GPUs.

    """
    count = ctypes.c_int()
    check_call(_LIB.MXGetGPUCount(ctypes.byref(count)))
    return count.value


def gpu_memory_info(device_id=0):
    """Query CUDA for the free and total bytes of GPU global memory.

    Parameters
    ----------
    device_id : int, optional
        The device id of the GPU device.

    Raises
    ------
    Will raise an exception on any CUDA error.

    Returns
    -------
    (free, total) : (int, int)
    """
    free = ctypes.c_uint64()
    total = ctypes.c_uint64()
    dev_id = ctypes.c_int(device_id)
    check_call(_LIB.MXGetGPUMemoryInformation64(dev_id, ctypes.byref(free), ctypes.byref(total)))
    return (free.value, total.value)


_current = contextvars.ContextVar('namemanager', default=Device('cpu', 0))


def current_device():
    """Returns the current device.

    By default, `mx.cpu()` is used for all the computations
    and it can be overridden by using `with mx.Device(x)` statement where
    x can be cpu(device_id) or gpu(device_id).

    Examples
    -------
    >>> mx.current_device()
    cpu(0)
    >>> with mx.Device('gpu', 1):  # Device changed in `with` block.
    ...    mx.current_device()  # Computation done here will be on gpu(1).
    ...
    gpu(1)
    >>> mx.current_device() # Back to default device.
    cpu(0)

    Returns
    -------
    default_device : Device
    """
    return _current.get()
