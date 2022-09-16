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
# pylint: disable=unnecessary-pass, too-many-lines
"""Neural network parameter."""

__all__ = ['DeferredInitializationError', 'Parameter', 'Constant',
           'tensor_types']


import uuid
import warnings
import weakref
import numpy as np

from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, device as _device, _deferred_compute as dc
from ..device import Device, cpu
from .. import autograd
from .utils import shape_is_known
from ..util import is_np_shape, is_np_array, wrap_ctx_to_device_func
from .. import numpy as _mx_np  # pylint: disable=reimported

# pylint: disable= invalid-name
tensor_types = (symbol.Symbol, ndarray.NDArray)
# pylint: enable= invalid-name

class DeferredInitializationError(MXNetError):
    """Error for unfinished deferred initialization."""
    pass

class Parameter(object):
    """A Container holding parameters (weights) of Blocks.

    :py:class:`Parameter` holds a copy of the parameter on each :py:class:`Device` after
    it is initialized with ``Parameter.initialize(...)``. If :py:attr:`grad_req` is
    not ``'null'``, it will also hold a gradient array on each :py:class:`Device`::

        device = mx.gpu(0)
        x = mx.np.zeros((16, 100), device=device)
        w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())
        b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Zero())
        w.initialize(device=device)
        b.initialize(device=device)
        out = mx.npx.fully_connected(x, w.data(device), b.data(device), num_hidden=64)

    Parameters
    ----------
    name : str, default 'weight'
        Name of this parameter. It decides the corresponding default initializer.
    grad_req : {'write', 'add', 'null'}, default 'write'
        Specifies how to update gradient to grad arrays.

        - ``'write'`` means everytime gradient is written to grad :py:class:`NDArray`.
        - ``'add'`` means everytime gradient is added to the grad :py:class:`NDArray`. You need
          to manually call ``zero_grad()`` to clear the gradient buffer before each
          iteration when using this option.
        - 'null' means gradient is not requested for this parameter. gradient arrays
          will not be allocated.
    shape : int or tuple of int, default None
        Shape of this parameter. By default shape is not specified. Parameter with
        unknown shape can be used for :py:class:`Symbol` API, but ``init`` will throw an error
        when using :py:class:`NDArray` API.
    dtype : numpy.dtype or str, default 'float32'
        Data type of this parameter. For example, ``numpy.float32`` or ``'float32'``.
    lr_mult : float, default 1.0
        Learning rate multiplier. Learning rate will be multiplied by lr_mult
        when updating this parameter with optimizer.
    wd_mult : float, default 1.0
        Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.
    init : Initializer, default None
        Initializer of this parameter. Will use the global initializer by default.
    stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter.
    grad_stype: {'default', 'row_sparse', 'csr'}, defaults to 'default'.
        The storage type of the parameter's gradient.

    Attributes
    ----------
    grad_req : {'write', 'add', 'null'}
        This can be set before or after initialization. Setting ``grad_req`` to ``'null'``
        with ``x.grad_req = 'null'`` saves memory and computation when you don't
        need gradient w.r.t x.
    lr_mult : float
        Local learning rate multiplier for this Parameter. The actual learning rate
        is calculated with ``learning_rate * lr_mult``. You can set it with
        ``param.lr_mult = 2.0``
    wd_mult : float
        Local weight decay multiplier for this Parameter.
    """
    def __init__(self, name='weight', grad_req='write', shape=None, dtype=mx_real_t,
                 lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False,
                 differentiable=True, stype='default', grad_stype='default'):
        self._var = None
        self._uuid = str(uuid.uuid4())
        self._var_name = None
        self._data = None
        self._grad = None
        self._device_list = None
        self._device_map = None
        self._trainer = None
        self._deferred_init = ()
        self._differentiable = differentiable
        self._allow_deferred_init = allow_deferred_init
        self._grad_req = None
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._name = name
        self._dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init
        # sparse related storage type information
        valid_stypes = ['default', 'row_sparse', 'csr']
        assert grad_stype in valid_stypes, "grad_stype for Parameter must be " \
            f"one of 'default', 'row_sparse', or 'csr', but got '{grad_stype}'"
        assert stype in valid_stypes, "stype for Parameter must be " \
            f"one of 'default', 'row_sparse', or 'csr', but got '{stype}'"
        self._grad_stype = grad_stype
        self._stype = stype

    def __repr__(self):
        s = 'Parameter (shape={shape}, dtype={dtype})'
        return s.format(shape=self.shape, dtype=self.dtype)

    @property
    def grad_req(self):
        return self._grad_req

    @property
    def name(self):
        return self._name

    @grad_req.setter
    def grad_req(self, req):
        assert req in ['write', 'add', 'null'], \
            f"grad_req must be one of 'write', 'add', or 'null', but got '{req}'"
        if not self._differentiable:
            req = 'null'
        if self._grad_req == req:
            return
        self._grad_req = req
        if req == 'null' and self._grad is not None:
            self._grad = None
            self._data = [i.detach() for i in self._data]
        elif self._data is not None:
            self._init_grad()

    @property
    def dtype(self):
        """The type of the parameter.

        Setting the dtype value is equivalent to casting the value of the parameter
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self.cast(dtype)

    @property
    def shape(self):
        """The shape of the parameter.

        By default, an unknown dimension size is 0. However, when the NumPy semantic
        is turned on, unknown dimension size is -1.
        """
        if self._shape is None:
            return None
        elif is_np_shape():
            # Parameters shouldn't be zero-size. If one of its dimension is 0,
            # it means the parameter isn't initialized. In the NumPy semantics,
            # the unknown dimension should be marked with -1.
            return tuple(i if i != 0 else -1 for i in self._shape)
        else:
            return self._shape

    @shape.setter
    def shape(self, new_shape):
        if self._shape is None:
            self._shape = new_shape
            return

        assert len(self._shape) == len(new_shape) and \
            all(j in (-1, 0, i) for i, j in zip(new_shape, self._shape)), \
            f"Expected shape {str(new_shape)} is incompatible with given shape {str(self._shape)} for Parameter {str(self.name)}." 
            # -1 means unknown dim size in np_shape mode

        self._shape = new_shape

    def _set_trainer(self, trainer):
        """ Set the trainer this parameter is associated with. """
        # trainer cannot be replaced for sparse params
        if self._stype != 'default' and self._trainer and trainer and self._trainer() is not trainer:
            raise RuntimeError(
                f"Failed to set the trainer for Parameter '{self.name}' because it was already set. " \
                f"More than one trainers for a {self._stype} Parameter is not supported.")
        if trainer is not None:
            self._trainer = weakref.ref(trainer)
        else:
            self._trainer = trainer

    def _check_and_get(self, arr_list, device):
        if arr_list is not None:
            if device is list:
                return arr_list
            if device is None:
                if len(arr_list) == 1:
                    return arr_list[0]
                else:
                    device = _device.current_device()
            device_list = self._device_map[device.device_typeid&1]
            if device.device_id < len(device_list):
                idx = device_list[device.device_id]
                if idx is not None:
                    return arr_list[idx]
            raise RuntimeError(
                f"Parameter '{self.name}' was not initialized on device {str(device)}. "
                f"It was only initialized on {str(self._device_list)}.")
        if self._deferred_init:
            raise DeferredInitializationError(
                f"Parameter '{self.name}' has not been initialized yet because initialization was " \
                "deferred. Actual initialization happens during the first forward pass. " \
                "Please pass one batch of data through the network before accessing Parameters. " \
                "You can also avoid deferred initialization by specifying in_units, " \
                "num_features, etc., for network layers.")
        raise RuntimeError(
            f"Parameter '{self.name}' has not been initialized. Note that " \
            "you should initialize parameters and create Trainer " \
            "with Block.collect_params() instead of Block.params " \
            "because the later does not include Parameters of " \
            "nested child Blocks")

    @wrap_ctx_to_device_func
    def _get_row_sparse(self, arr_list, device, row_id):
        """ Get row_sparse data from row_sparse parameters based on row_id. """
        # get row sparse params based on row ids
        if not isinstance(row_id, ndarray.NDArray):
            raise TypeError(f"row_id must have NDArray type, but {type(row_id)} is given")
        trainer = self._trainer() if self._trainer else None
        if not trainer:
            raise RuntimeError(f"Cannot get row_sparse data for Parameter '{self.name}' when no " \
                               "Trainer is created with it.")
        results = self._check_and_get(arr_list, device)

        # fetch row sparse params from the trainer
        trainer._row_sparse_pull(self, results, row_id)
        return results

    @wrap_ctx_to_device_func
    def _load_init(self, data, device, cast_dtype=False, dtype_source='current'):
        """
        (Re)initializes by loading from data.
        Parameters
        ----------
        data : NDArray
            The data to load
        device : Device or list of Device
            Device(s) initialize loaded parameters on.
        cast_dtype : bool, default False
            Cast the data type of the parameter
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        """
        if cast_dtype:
            assert dtype_source in ['current', 'saved']
        if self.shape:
            unknown_dim_size = -1 if is_np_shape() else 0
            for self_dim, data_dim in zip(self.shape, data.shape):
                assert self_dim in (unknown_dim_size, data_dim), \
                    f"Failed loading Parameter '{self.name}' from saved params: " \
                    f"shape incompatible expected {str(self.shape)} vs saved {str(data.shape)}"
            self.shape = tuple(i if i != unknown_dim_size else j
                               for i, j in zip(self.shape, data.shape))
        if self.dtype:
            if cast_dtype and self.dtype != data.dtype:
                if dtype_source == 'current':
                    data = data.astype(self.dtype, copy=False)
                elif dtype_source == 'saved':
                    self.dtype = data.dtype
            else:
                assert self.dtype == data.dtype, \
                f"Failed loading Parameter '{self.name}' from saved params: " \
                f"dtype incompatible expected {str(self.dtype)} vs saved {str(data.dtype)}. " \
                "Set cast_dtype=True to cast the dtype of saved params."
        if self._stype != data.stype:
            data = data.tostype(self._stype)
        if isinstance(device, Device):
            device = [device]
        if self._data is None:
            if self._deferred_init:
                assert device is None or set(device) == set(self._deferred_init[1]), \
                    f"Failed to load Parameter '{self.name}' on {str(device)} because it was " \
                    f"previous initialized on {str(self.list_device())}."
                device = self._deferred_init[1]
            elif device is None:
                device = [cpu()]
            self._init_impl(data, device)
        else:
            assert device is None or set(device) == set(self.list_device()), \
                f"Failed to load Parameter '{self.name}' on {str(device)} because it was " \
                f"previous initialized on {str(self.list_device())}."
            self.set_data(data)
        self._deferred_init = ()

    def _finish_deferred_init(self):
        """Finishes deferred initialization."""
        if not self._deferred_init:
            return
        init, device, default_init, data = self._deferred_init
        self._deferred_init = ()

        assert shape_is_known(self.shape), \
            f"Cannot initialize Parameter '{self.name}' because it has " \
            f"invalid shape: {str(self.shape)}. Please specify in_units, " \
            "in_channels, etc for `Block`s."

        with autograd.pause(), dc.context(False):
            if data is None:
                if is_np_array():
                    kwargs = {'shape': self.shape, 'dtype': self.dtype, 'device': cpu()}
                    if self._stype != 'default':
                        raise ValueError("Currently stype {} is not supported in NumPy interface and Gluon2.0"
                                         .format(self._stype))
                    zeros_fn = _mx_np.zeros
                else:
                    kwargs = {'shape': self.shape, 'dtype': self.dtype, 'ctx': cpu()}
                    kwargs['stype'] = self._stype
                    zeros_fn = ndarray.zeros
                data = zeros_fn(**kwargs)
                initializer.create(default_init)(
                    initializer.InitDesc(self.name, {'__init__': init}), data)

            self._init_impl(data, device)

    def _init_impl(self, data, device_list):
        """Sets data and grad."""
        self._device_list = list(device_list)
        self._device_map = [[], []]
        for i, device in enumerate(self._device_list):
            dev_list = self._device_map[device.device_typeid&1]
            while len(dev_list) <= device.device_id:
                dev_list.append(None)
            dev_list[device.device_id] = i

        self._data = [data.copyto(device) for device in self._device_list]
        self._init_grad()

    def _init_grad(self):
        """Initialize grad buffers."""
        if self.grad_req == 'null':
            self._grad = None
            return

        if is_np_array():
            if self._grad_stype != 'default':
                raise ValueError("Currently stype {} is not supported in NumPy interface and Gluon2.0"
                                 .format(self._grad_stype))
            self._grad = [_mx_np.zeros(shape=i.shape, dtype=i.dtype, device=i.device)
                          for i in self._data]
        else:
            self._grad = [ndarray.zeros(shape=i.shape, dtype=i.dtype, ctx=i.context,
                                        stype=self._grad_stype) for i in self._data]

        autograd.mark_variables(self._check_and_get(self._data, list),
                                self._grad, self.grad_req)

    def _reduce(self):
        """Reduce data from multiple device to cpu."""
        device = cpu()
        if self._stype == 'default':
            block = self.list_data()
            if len(block) > 1:
                if is_np_array():
                    data = sum([w.copyto(device) for w in block]) / len(block)
                else:
                    data = ndarray.add_n(*(w.copyto(device) for w in block)) / len(block)
            else:
                data = self.data().copyto(device)
        else:
            # fetch all rows for 'row_sparse' param
            all_row_ids = ndarray.arange(0, self.shape[0], dtype='int64', ctx=device)
            data = ndarray.zeros(self.shape, stype='row_sparse', ctx=device)
            trainer = self._trainer() if self._trainer else None
            if not trainer:
                raise RuntimeError(f"Cannot reduce row_sparse data for Parameter '{self.name}' when no " \
                                   "Trainer is created with it.")
            trainer._row_sparse_pull(self, data, all_row_ids, full_idx=True)
        return data

    @wrap_ctx_to_device_func
    def initialize(self, init=None, device=None, default_init=initializer.Uniform(),
                   force_reinit=False):
        """Initializes parameter and gradient arrays. Only used for :py:class:`NDArray` API.

        Parameters
        ----------
        init : Initializer
            The initializer to use. Overrides :py:meth:`Parameter.init` and default_init.
        device : Device or list of Device, default :py:meth:`device.current_device()`.
            Assign Parameter to given device. If device is a list of Device, a
            copy will be made for each device.

            .. note::
                Copies are independent arrays. User is responsible for keeping
                their values consistent when updating.
                Normally :py:class:`gluon.Trainer` does this for you.

        default_init : Initializer
            Default initializer is used when both :py:func:`init`
            and :py:meth:`Parameter.init` are ``None``.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        Examples
        --------
        >>> weight = mx.gluon.Parameter('weight', shape=(2, 2))
        >>> weight.initialize(device=mx.cpu(0))
        >>> weight.data()
        [[-0.01068833  0.01729892]
         [ 0.02042518 -0.01618656]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.grad()
        [[ 0.  0.]
         [ 0.  0.]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.initialize(device=[mx.gpu(0), mx.gpu(1)])
        >>> weight.data(mx.gpu(0))
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(0)>
        >>> weight.data(mx.gpu(1))
        [[-0.00873779 -0.02834515]
         [ 0.05484822 -0.06206018]]
        <NDArray 2x2 @gpu(1)>
        """
        if self._data is not None and not force_reinit:
            warnings.warn(f"Parameter '{self.name}' is already initialized, ignoring. " \
                          "Set force_reinit=True to re-initialize.",
                          stacklevel=2)
            return
        self._data = self._grad = None
        if device is None:
            device = [_device.current_device()]
        if isinstance(device, Device):
            device = [device]
        if isinstance(self.init, initializer.RNNFused):
            self.init.set_initializer(init if init else default_init)
            init = default_init = self.init
        if init is None:
            init = default_init if self.init is None else self.init
        if not shape_is_known(self.shape):
            if self._allow_deferred_init:
                self._deferred_init = (init, device, default_init, None)
                return
            raise ValueError(f"Cannot initialize Parameter '{self.name}' because it has " \
                             f"invalid shape: {str(self.shape)}.")

        self._deferred_init = (init, device, default_init, None)
        self._finish_deferred_init()

    def reset_device(self, device):
        """Re-assign Parameter to other devices.

        Parameters
        ----------
        device : Device or list of Device, default ``device.current_device()``.
            Assign Parameter to given device. If device is a list of Device, a
            copy will be made for each device.
        """
        if device is None:
            device = [_device.current_device()]
        if isinstance(device, Device):
            device = [device]
        if self._data:
            data = self._reduce()
            with autograd.pause():
                self._init_impl(data, device)
        elif self._deferred_init:
            init, _, default_init, data = self._deferred_init
            self._deferred_init = (init, device, default_init, data)
        else:
            raise ValueError(f"Cannot reset device for Parameter '{self.name}' because it "
                             "has not been initialized.")

    def reset_ctx(self, ctx):
        """This function has been deprecated. Please refer to ``Parameter.reset_device``."""
        warnings.warn('Parameter.reset_ctx has been renamed to'
                      ' Parameter.reset_device', DeprecationWarning)
        self.reset_device(ctx)

    def set_data(self, data):
        """Sets this parameter's value on all devices."""
        self.shape = data.shape

        if self._data is None:
            assert self._deferred_init, \
                f"Parameter '{self.name}' has not been initialized"
            self._deferred_init = self._deferred_init[:3] + (data,)
            return

        # if update_on_kvstore, we need to make sure the copy stored in kvstore is in sync
        trainer = self._trainer() if self._trainer else None
        if trainer and trainer._kv_initialized and trainer._update_on_kvstore:
            if self not in trainer._params_to_init:
                trainer._reset_kvstore()

        for arr in self._check_and_get(self._data, list):
            arr[:] = data

    def row_sparse_data(self, row_id):
        """Returns a copy of the 'row_sparse' parameter on the same device as row_id's.
        The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized on this device before.

        Parameters
        ----------
        row_id: NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        NDArray on row_id's device
        """
        if self._stype != 'row_sparse':
            raise RuntimeError(f"Cannot return a copy of Parameter {self.name} via row_sparse_data() " \
                               f"because its storage type is {self._stype}. Please use data() instead.")
        return self._get_row_sparse(self._data, row_id.device, row_id)

    def list_row_sparse_data(self, row_id):
        """Returns copies of the 'row_sparse' parameter on all devices, in the same order
        as creation. The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized before.

        Parameters
        ----------
        row_id: NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        list of NDArrays
        """
        if self._stype != 'row_sparse':
            raise RuntimeError(f"Cannot return copies of Parameter '{self.name}' on all devices via " \
                               f"list_row_sparse_data() because its storage type is {self._stype}. Please " \
                               "use data() instead.")
        return self._get_row_sparse(self._data, list, row_id)

    @wrap_ctx_to_device_func
    def data(self, device=None):
        """Returns a copy of this parameter on one device. Must have been
        initialized on this device before. For sparse parameters, use
        :py:meth:`Parameter.row_sparse_data` instead.

        Parameters
        ----------
        device : Device
            Desired device.

        Returns
        -------
        NDArray on device
        """
        if self._stype != 'default':
            raise RuntimeError(f"Cannot return a copy of Parameter '{self.name}' on device {str(device)} via data() " \
                               f"because its storage type is {self._stype}. Please use row_sparse_data() instead.")
        data = self._check_and_get(self._data, device)
        dc.set_variable(data, self.var())
        return data

    def list_data(self):
        """Returns copies of this parameter on all devices, in the same order
        as creation. For sparse parameters, use :py:meth:`Parameter.list_row_sparse_data`
        instead.

        Returns
        -------
        list of NDArrays
        """
        if self._stype != 'default':
            raise RuntimeError(f"Cannot return copies of Parameter '{self.name}' on all devices via " \
                               f"list_data() because its storage type is {self._stype}. Please use " \
                               "row_sparse_data() instead.")
        return self._check_and_get(self._data, list)

    def grad(self, device=None):
        """Returns a gradient buffer for this parameter on one device.

        Parameters
        ----------
        device : Device
            Desired device.
        """
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                f"Cannot get gradient array for Parameter '{self.name}' " \
                "because grad_req='null'")
        return self._check_and_get(self._grad, device)

    def list_grad(self):
        """Returns gradient buffers on all devices, in the same order
        as :py:meth:`values`."""
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                f"Cannot get gradient array for Parameter '{self.name}' " \
                "because grad_req='null'")
        return self._check_and_get(self._grad, list)

    def list_ctx(self):
        """This function has been deprecated. Please refer to ``Parameter.list_device``."""
        warnings.warn('Parameter.list_ctx has been renamed to'
                      ' Parameter.list_device', DeprecationWarning)
        return self.list_device()

    def list_device(self):
        """Returns a list of devices this parameter is initialized on."""
        if self._data is None:
            if self._deferred_init:
                return self._deferred_init[1]
            raise RuntimeError(f"Parameter '{self.name}' has not been initialized")
        return self._device_list

    def zero_grad(self):
        """Sets gradient buffer on all devices to 0. No action is taken if
        parameter is uninitialized or doesn't require gradient."""
        if self._grad is None:
            return
        for i in self._grad:
            ndarray.zeros_like(i, out=i)

    def var(self):
        """Returns a symbol representing this parameter."""
        if self._var is None:
            if self._var_name is None:  # _var_name is set manually in SymbolBlock.import
                # The variable name is required by the storage profiler.
                self._var_name = self._uuid.replace('-', '_') + '_' + self._name
            self._var = symbol.var(self._var_name, shape=self.shape, dtype=self.dtype,
                                   lr_mult=self.lr_mult, wd_mult=self.wd_mult,
                                   init=self.init, stype=self._stype)
            if is_np_array():
                self._var = self._var.as_np_ndarray()
        return self._var

    def cast(self, dtype):
        """Cast data and gradient of this Parameter to a new data type.

        Parameters
        ----------
        dtype : str or numpy.dtype
            The new data type.
        """
        self._dtype = dtype
        self._var = None  # Clear Symbol Variable as it caches the dtype
        if self._data is None:
            return
        with autograd.pause():
            self._data = [i.astype(dtype) for i in self._data]
            if self._grad is None:
                return
            self._grad = [i.astype(dtype) for i in self._grad]
            autograd.mark_variables(self._data, self._grad, self.grad_req)

    def _check_and_setattr(self, **kwargs):
        """check and set attributes for parameter"""
        for k, v in kwargs.items():
            if hasattr(self, k) and getattr(self, k) is not None:
                existing = getattr(self, k)
                if k == 'shape' and len(v) == len(existing):
                    inferred_shape = []
                    matched = True
                    for dim1, dim2 in zip(v, existing):
                        if dim1 != dim2 and dim1 > 0 and dim2 > 0:
                            matched = False
                            break
                        elif dim1 == dim2:
                            inferred_shape.append(dim1)
                        elif dim1 in (0, -1):  # -1 means unknown dim size in np_shape mode
                            inferred_shape.append(dim2)
                        else:
                            inferred_shape.append(dim1)

                    if matched:
                        self._shape = tuple(inferred_shape)
                        continue
                elif k == 'dtype' and np.dtype(v) == np.dtype(existing):
                    continue

                assert v is None or v == existing, \
                    f"Cannot retrieve Parameter '{self.name}' because desired attribute " \
                    f"does not match with stored for attribute '{k}': " \
                    f"desired '{str(v)}' vs stored '{str(getattr(self, k))}'."
            else:
                setattr(self, k, v)

class Constant(Parameter):
    """A constant parameter for holding immutable tensors.
    `Constant`s are ignored by `autograd` and `Trainer`, thus their values
    will not change during training. But you can still update their values
    manually with the `set_data` method.

    `Constant` s can be created with either::

        const = mx.gluon.Constant([[1,2],[3,4]])

    or::

        class Block(gluon.Block):
            def __init__(self, **kwargs):
                super(Block, self).__init__(**kwargs)
                self.const = mx.gluon.Constant([[1,2],[3,4]])

    Parameters
    ----------
    value : array-like
        Initial value for the constant.
    """
    def __init__(self, value):
        if not isinstance(value, ndarray.NDArray):
            array_fn = _mx_np.array if is_np_array() else ndarray.array
            value = array_fn(value)
        self.value = value

        class Init(initializer.Initializer):
            def _init_weight(self, _, arr):
                value.copyto(arr)
        init_name = 'Constant_{}'.format(id(self))
        initializer.alias(init_name)(Init)

        super(Constant, self).__init__(
            name='const', grad_req='null', shape=value.shape, dtype=value.dtype,
            init=init_name)

    def __repr__(self):
        s = 'Constant (shape={shape}, dtype={dtype})'
        return s.format(shape=self.shape, dtype=self.dtype)

    @property
    def grad_req(self):
        return 'null'

    @grad_req.setter
    def grad_req(self, req):
        if req != 'null':
            warnings.warn('Constant parameter "{}" does not support '
                          'grad_req other than "null", and new value "{}" '
                          'is ignored.'.format(self.name, req))
