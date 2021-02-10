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
           'ParameterDict', 'tensor_types']


from collections import OrderedDict, defaultdict
import warnings
import numpy as np

from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported

# pylint: disable= invalid-name
tensor_types = (symbol.Symbol, ndarray.NDArray)
# pylint: enable= invalid-name

class DeferredInitializationError(MXNetError):
    """Error for unfinished deferred initialization."""
    pass

class Parameter(object):
    """A Container holding parameters (weights) of Blocks.

    :py:class:`Parameter` holds a copy of the parameter on each :py:class:`Context` after
    it is initialized with ``Parameter.initialize(...)``. If :py:attr:`grad_req` is
    not ``'null'``, it will also hold a gradient array on each :py:class:`Context`::

        ctx = mx.gpu(0)
        x = mx.nd.zeros((16, 100), ctx=ctx)
        w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())
        b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Zero())
        w.initialize(ctx=ctx)
        b.initialize(ctx=ctx)
        out = mx.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)

    Parameters
    ----------
    name : str
        Name of this parameter.
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
    def __init__(self, name, grad_req='write', shape=None, dtype=mx_real_t,
                 lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False,
                 differentiable=True, stype='default', grad_stype='default'):
        self._var = None
        self._data = None
        self._grad = None
        self._ctx_list = None
        self._ctx_map = None
        self._trainer = None
        self._deferred_init = ()
        self._differentiable = differentiable
        self._allow_deferred_init = allow_deferred_init
        self._grad_req = None
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self.name = name
        self._dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init
        # sparse related storage type information
        valid_stypes = ['default', 'row_sparse', 'csr']
        assert grad_stype in valid_stypes, "grad_stype for Parameter '%s' must be " \
            "one of 'default', 'row_sparse', or 'csr', but got '%s'" % (name, grad_stype)
        assert stype in valid_stypes, "stype for Parameter '%s' must be " \
            "one of 'default', 'row_sparse', or 'csr', but got '%s'" % (name, stype)
        self._grad_stype = grad_stype
        self._stype = stype

    def __repr__(self):
        s = 'Parameter {name} (shape={shape}, dtype={dtype})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype)

    @property
    def grad_req(self):
        return self._grad_req

    @grad_req.setter
    def grad_req(self, req):
        assert req in ['write', 'add', 'null'], \
            "grad_req must be one of 'write', 'add', or 'null', but got '%s'"%req
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
            "Expected shape %s is incompatible with given shape %s."%(
                str(new_shape), str(self._shape))  # -1 means unknown dim size in np_shape mode

        self._shape = new_shape

    def _set_trainer(self, trainer):
        """ Set the trainer this parameter is associated with. """
        # trainer cannot be replaced for sparse params
        if self._stype != 'default' and self._trainer and trainer and self._trainer is not trainer:
            raise RuntimeError(
                "Failed to set the trainer for Parameter '%s' because it was already set. " \
                "More than one trainers for a %s Parameter is not supported." \
                %(self.name, self._stype))
        self._trainer = trainer

    def _check_and_get(self, arr_list, ctx):
        if arr_list is not None:
            if ctx is list:
                return arr_list
            if ctx is None:
                if len(arr_list) == 1:
                    return arr_list[0]
                else:
                    ctx = context.current_context()
            ctx_list = self._ctx_map[ctx.device_typeid&1]
            if ctx.device_id < len(ctx_list):
                idx = ctx_list[ctx.device_id]
                if idx is not None:
                    return arr_list[idx]
            raise RuntimeError(
                "Parameter '%s' was not initialized on context %s. "
                "It was only initialized on %s."%(
                    self.name, str(ctx), str(self._ctx_list)))
        if self._deferred_init:
            raise DeferredInitializationError(
                "Parameter '%s' has not been initialized yet because initialization was " \
                "deferred. Actual initialization happens during the first forward pass. " \
                "Please pass one batch of data through the network before accessing Parameters. " \
                "You can also avoid deferred initialization by specifying in_units, " \
                "num_features, etc., for network layers."%(self.name))
        raise RuntimeError(
            "Parameter '%s' has not been initialized. Note that " \
            "you should initialize parameters and create Trainer " \
            "with Block.collect_params() instead of Block.params " \
            "because the later does not include Parameters of " \
            "nested child Blocks"%(self.name))

    def _get_row_sparse(self, arr_list, ctx, row_id):
        """ Get row_sparse data from row_sparse parameters based on row_id. """
        # get row sparse params based on row ids
        if not isinstance(row_id, ndarray.NDArray):
            raise TypeError("row_id must have NDArray type, but %s is given"%(type(row_id)))
        if not self._trainer:
            raise RuntimeError("Cannot get row_sparse data for Parameter '%s' when no " \
                               "Trainer is created with it."%self.name)
        results = self._check_and_get(arr_list, ctx)

        # fetch row sparse params from the trainer
        self._trainer._row_sparse_pull(self, results, row_id)
        return results

    def _load_init(self, data, ctx, cast_dtype=False, dtype_source='current'):
        """
        (Re)initializes by loading from data.
        Parameters
        ----------
        data : NDArray
            The data to load
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
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
                    "Failed loading Parameter '%s' from saved params: " \
                    "shape incompatible expected %s vs saved %s"%(
                        self.name, str(self.shape), str(data.shape))
            self.shape = tuple(i if i != unknown_dim_size else j
                               for i, j in zip(self.shape, data.shape))
        if self.dtype:
            if cast_dtype and np.dtype(self.dtype).type != data.dtype:
                if dtype_source == 'current':
                    data = data.astype(self.dtype, copy=False)
                elif dtype_source == 'saved':
                    self.dtype = data.dtype
            else:
                if data.dtype == np.dtype([('bfloat16', np.uint16)]):
                    assert np.dtype(self.dtype) == data.dtype, \
                    "Failed loading Parameter '%s' from saved params: " \
                    "dtype incompatible expected %s vs saved %s. " \
                    "Set cast_dtype=True to cast the dtype of saved params."%(
                        self.name, str(self.dtype), str(data.dtype))
                else:
                    assert np.dtype(self.dtype).type == data.dtype, \
                    "Failed loading Parameter '%s' from saved params: " \
                    "dtype incompatible expected %s vs saved %s. " \
                    "Set cast_dtype=True to cast the dtype of saved params."%(
                        self.name, str(self.dtype), str(data.dtype))
        if self._stype != data.stype:
            data = data.tostype(self._stype)
        if isinstance(ctx, Context):
            ctx = [ctx]
        if self._data is None:
            if self._deferred_init:
                assert ctx is None or set(ctx) == set(self._deferred_init[1]), \
                    "Failed to load Parameter '%s' on %s because it was " \
                    "previous initialized on %s."%(
                        self.name, str(ctx), str(self.list_ctx()))
                ctx = self._deferred_init[1]
            elif ctx is None:
                ctx = [cpu()]
            self._init_impl(data, ctx)
        else:
            assert ctx is None or set(ctx) == set(self.list_ctx()), \
                "Failed to load Parameter '%s' on %s because it was " \
                "previous initialized on %s."%(
                    self.name, str(ctx), str(self.list_ctx()))
            self.set_data(data)
        self._deferred_init = ()

    def _finish_deferred_init(self):
        """Finishes deferred initialization."""
        if not self._deferred_init:
            return
        init, ctx, default_init, data = self._deferred_init
        self._deferred_init = ()

        assert shape_is_known(self.shape), \
            "Cannot initialize Parameter '%s' because it has " \
            "invalid shape: %s. Please specify in_units, " \
            "in_channels, etc for `Block`s."%(
                self.name, str(self.shape))

        with autograd.pause():
            if data is None:
                kwargs = {'shape': self.shape, 'dtype': self.dtype, 'ctx': context.cpu()}
                if is_np_array():
                    if self._stype != 'default':
                        raise ValueError("mxnet.numpy.zeros does not support stype = {}"
                                         .format(self._stype))
                    zeros_fn = _mx_np.zeros
                else:
                    kwargs['stype'] = self._stype
                    zeros_fn = ndarray.zeros
                data = zeros_fn(**kwargs)
                initializer.create(default_init)(
                    initializer.InitDesc(self.name, {'__init__': init}), data)

            self._init_impl(data, ctx)

    def _init_impl(self, data, ctx_list):
        """Sets data and grad."""
        self._ctx_list = list(ctx_list)
        self._ctx_map = [[], []]
        for i, ctx in enumerate(self._ctx_list):
            dev_list = self._ctx_map[ctx.device_typeid&1]
            while len(dev_list) <= ctx.device_id:
                dev_list.append(None)
            dev_list[ctx.device_id] = i

        self._data = [data.copyto(ctx) for ctx in self._ctx_list]
        self._init_grad()

    def _init_grad(self):
        """Initialize grad buffers."""
        if self.grad_req == 'null':
            self._grad = None
            return

        if is_np_array():
            if self._grad_stype != 'default':
                raise ValueError("mxnet.numpy.zeros does not support stype = {}"
                                 .format(self._grad_stype))
            self._grad = [_mx_np.zeros(shape=i.shape, dtype=i.dtype, ctx=i.ctx)
                          for i in self._data]
        else:
            self._grad = [ndarray.zeros(shape=i.shape, dtype=i.dtype, ctx=i.ctx,
                                        stype=self._grad_stype) for i in self._data]

        autograd.mark_variables(self._check_and_get(self._data, list),
                                self._grad, self.grad_req)

    def _reduce(self):
        """Reduce data from multiple context to cpu."""
        ctx = context.cpu()
        if self._stype == 'default':
            block = self.list_data()
            if len(block) > 1:
                if is_np_array():
                    data = sum([w.copyto(ctx) for w in block]) / len(block)
                else:
                    data = ndarray.add_n(*(w.copyto(ctx) for w in block)) / len(block)
            else:
                data = self.data().copyto(ctx)
        else:
            # fetch all rows for 'row_sparse' param
            all_row_ids = ndarray.arange(0, self.shape[0], dtype='int64', ctx=ctx)
            data = ndarray.zeros(self.shape, stype='row_sparse', ctx=ctx)
            self._trainer._row_sparse_pull(self, data, all_row_ids, full_idx=True)
        return data

    def initialize(self, init=None, ctx=None, default_init=initializer.Uniform(),
                   force_reinit=False):
        """Initializes parameter and gradient arrays. Only used for :py:class:`NDArray` API.

        Parameters
        ----------
        init : Initializer
            The initializer to use. Overrides :py:meth:`Parameter.init` and default_init.
        ctx : Context or list of Context, defaults to :py:meth:`context.current_context()`.
            Initialize Parameter on given context. If ctx is a list of Context, a
            copy will be made for each context.

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
        >>> weight.initialize(ctx=mx.cpu(0))
        >>> weight.data()
        [[-0.01068833  0.01729892]
         [ 0.02042518 -0.01618656]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.grad()
        [[ 0.  0.]
         [ 0.  0.]]
        <NDArray 2x2 @cpu(0)>
        >>> weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])
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
            warnings.warn("Parameter '%s' is already initialized, ignoring. " \
                          "Set force_reinit=True to re-initialize."%self.name,
                          stacklevel=2)
            return
        self._data = self._grad = None

        if ctx is None:
            ctx = [context.current_context()]
        if isinstance(ctx, Context):
            ctx = [ctx]
        if init is None:
            init = default_init if self.init is None else self.init
        if not shape_is_known(self.shape):
            if self._allow_deferred_init:
                self._deferred_init = (init, ctx, default_init, None)
                return
            raise ValueError("Cannot initialize Parameter '%s' because it has " \
                             "invalid shape: %s."%(self.name, str(self.shape)))

        self._deferred_init = (init, ctx, default_init, None)
        self._finish_deferred_init()

    def reset_ctx(self, ctx):
        """Re-assign Parameter to other contexts.

        Parameters
        ----------
        ctx : Context or list of Context, default ``context.current_context()``.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        if ctx is None:
            ctx = [context.current_context()]
        if isinstance(ctx, Context):
            ctx = [ctx]
        if self._data:
            data = self._reduce()
            with autograd.pause():
                self._init_impl(data, ctx)
        elif self._deferred_init:
            init, _, default_init, data = self._deferred_init
            self._deferred_init = (init, ctx, default_init, data)
        else:
            raise ValueError("Cannot reset context for Parameter '%s' because it "
                             "has not been initialized."%self.name)

    def set_data(self, data):
        """Sets this parameter's value on all contexts."""
        self.shape = data.shape

        if self._data is None:
            assert self._deferred_init, \
                "Parameter '%s' has not been initialized"%self.name
            self._deferred_init = self._deferred_init[:3] + (data,)
            return

        # if update_on_kvstore, we need to make sure the copy stored in kvstore is in sync
        if self._trainer and self._trainer._kv_initialized and self._trainer._update_on_kvstore:
            if self not in self._trainer._params_to_init:
                self._trainer._reset_kvstore()

        for arr in self._check_and_get(self._data, list):
            arr[:] = data

    def row_sparse_data(self, row_id):
        """Returns a copy of the 'row_sparse' parameter on the same context as row_id's.
        The copy only retains rows whose ids occur in provided row ids.
        The parameter must have been initialized on this context before.

        Parameters
        ----------
        row_id: NDArray
            Row ids to retain for the 'row_sparse' parameter.

        Returns
        -------
        NDArray on row_id's context
        """
        if self._stype != 'row_sparse':
            raise RuntimeError("Cannot return a copy of Parameter %s via row_sparse_data() " \
                               "because its storage type is %s. Please use data() instead." \
                               %(self.name, self._stype))
        return self._get_row_sparse(self._data, row_id.ctx, row_id)

    def list_row_sparse_data(self, row_id):
        """Returns copies of the 'row_sparse' parameter on all contexts, in the same order
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
            raise RuntimeError("Cannot return copies of Parameter '%s' on all contexts via " \
                               "list_row_sparse_data() because its storage type is %s. Please " \
                               "use data() instead." % (self.name, self._stype))
        return self._get_row_sparse(self._data, list, row_id)

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before. For sparse parameters, use
        :py:meth:`Parameter.row_sparse_data` instead.

        Parameters
        ----------
        ctx : Context
            Desired context.

        Returns
        -------
        NDArray on ctx
        """
        if self._stype != 'default':
            raise RuntimeError("Cannot return a copy of Parameter '%s' on ctx %s via data() " \
                               "because its storage type is %s. Please use row_sparse_data() " \
                               "instead." % (self.name, str(ctx), self._stype))
        return self._check_and_get(self._data, ctx)

    def list_data(self):
        """Returns copies of this parameter on all contexts, in the same order
        as creation. For sparse parameters, use :py:meth:`Parameter.list_row_sparse_data`
        instead.

        Returns
        -------
        list of NDArrays
        """
        if self._stype != 'default':
            raise RuntimeError("Cannot return copies of Parameter '%s' on all contexts via " \
                               "list_data() because its storage type is %s. Please use " \
                               "row_sparse_data() instead." % (self.name, self._stype))
        return self._check_and_get(self._data, list)

    def grad(self, ctx=None):
        """Returns a gradient buffer for this parameter on one context.

        Parameters
        ----------
        ctx : Context
            Desired context.
        """
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter '%s' " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, ctx)

    def list_grad(self):
        """Returns gradient buffers on all contexts, in the same order
        as :py:meth:`values`."""
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter '%s' " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, list)

    def list_ctx(self):
        """Returns a list of contexts this parameter is initialized on."""
        if self._data is None:
            if self._deferred_init:
                return self._deferred_init[1]
            raise RuntimeError("Parameter '%s' has not been initialized"%self.name)
        return self._ctx_list

    def zero_grad(self):
        """Sets gradient buffer on all contexts to 0. No action is taken if
        parameter is uninitialized or doesn't require gradient."""
        if self._grad is None:
            return
        for i in self._grad:
            ndarray.zeros_like(i, out=i)

    def var(self):
        """Returns a symbol representing this parameter."""
        if self._var is None:
            self._var = symbol.var(self.name, shape=self.shape, dtype=self.dtype,
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
        if self._data is None:
            return
        with autograd.pause():
            self._data = [i.astype(dtype) for i in self._data]
            if self._grad is None:
                return
            self._grad = [i.astype(dtype) for i in self._grad]
            autograd.mark_variables(self._data, self._grad, self.grad_req)


class Constant(Parameter):
    """A constant parameter for holding immutable tensors.
    `Constant`s are ignored by `autograd` and `Trainer`, thus their values
    will not change during training. But you can still update their values
    manually with the `set_data` method.

    `Constant` s can be created with either::

        const = mx.gluon.Constant('const', [[1,2],[3,4]])

    or::

        class Block(gluon.Block):
            def __init__(self, **kwargs):
                super(Block, self).__init__(**kwargs)
                self.const = self.params.get_constant('const', [[1,2],[3,4]])

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : array-like
        Initial value for the constant.
    """
    def __init__(self, name, value):
        if not isinstance(value, ndarray.NDArray):
            array_fn = _mx_np.array if is_np_array() else ndarray.array
            value = array_fn(value)
        self.value = value

        class Init(initializer.Initializer):
            def _init_weight(self, _, arr):
                value.copyto(arr)
        init_name = 'Constant_{}_{}'.format(name, id(self))
        initializer.alias(init_name)(Init)

        super(Constant, self).__init__(
            name, grad_req='null', shape=value.shape, dtype=value.dtype,
            init=init_name)

    def __repr__(self):
        s = 'Constant {name} (shape={shape}, dtype={dtype})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype)

    @property
    def grad_req(self):
        return 'null'

    @grad_req.setter
    def grad_req(self, req):
        if req != 'null':
            warnings.warn('Constant parameter "{}" does not support '
                          'grad_req other than "null", and new value "{}" '
                          'is ignored.'.format(self.name, req))


class ParameterDict(object):
    """A dictionary managing a set of parameters.

    Parameters
    ----------
    prefix : str, default ``''``
        The prefix to be prepended to all Parameters' names created by this dict.
    shared : ParameterDict or None
        If not ``None``, when this dict's :py:meth:`get` method creates a new parameter, will
        first try to retrieve it from "shared" dict. Usually used for sharing
        parameters with another Block.
    """
    def __init__(self, prefix='', shared=None):
        self._prefix = prefix
        self._params = OrderedDict()
        self._shared = shared

    def __repr__(self):
        s = '{name}(\n{content}\n)'
        name = self._prefix+' ' if self._prefix else ''
        return s.format(name=name,
                        content='\n'.join([_indent('  {0}'.format(v), 2)
                                           for v in self.values()]))

    def __getitem__(self, key):
        return self._params[key]

    def __iter__(self):
        return iter(self._params)

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    @property
    def prefix(self):
        """Prefix of this dict. It will be prepended to :py:class:`Parameter`s' name created
        with :py:func:`get`."""
        return self._prefix

    def _get_impl(self, name):
        if name in self._params:
            return self._params[name]
        if self._shared is not None and name in self._shared._params:
            self._params[name] = self._shared._params[name]
            return self._shared._params[name]
        return None

    def get(self, name, **kwargs):
        """Retrieves a :py:class:`Parameter` with name ``self.prefix+name``. If not found,
        :py:func:`get` will first try to retrieve it from "shared" dict. If still not
        found, :py:func:`get` will create a new :py:class:`Parameter` with key-word arguments and
        insert it to self.

        Parameters
        ----------
        name : str
            Name of the desired Parameter. It will be prepended with this dictionary's
            prefix.
        **kwargs : dict
            The rest of key-word arguments for the created :py:class:`Parameter`.

        Returns
        -------
        Parameter
            The created or retrieved :py:class:`Parameter`.
        """
        name = self.prefix + name
        param = self._get_impl(name)
        if param is None: # pylint: disable=too-many-nested-blocks
            param = Parameter(name, **kwargs)
            self._params[name] = param
        else:
            for k, v in kwargs.items():
                if hasattr(param, k) and getattr(param, k) is not None:
                    existing = getattr(param, k)
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
                            param._shape = tuple(inferred_shape)
                            continue
                    elif k == 'dtype' and np.dtype(v) == np.dtype(existing):
                        continue

                    assert v is None or v == existing, \
                        "Cannot retrieve Parameter '%s' because desired attribute " \
                        "does not match with stored for attribute '%s': " \
                        "desired '%s' vs stored '%s'."%(
                            name, k, str(v), str(getattr(param, k)))
                else:
                    setattr(param, k, v)
        return param

    def get_constant(self, name, value=None):
        """Retrieves a :py:class:`.Constant` with name ``self.prefix+name``. If not found,
        :py:func:`get` will first try to retrieve it from "shared" dict. If still not
        found, :py:func:`get` will create a new :py:class:`.Constant` with key-word
        arguments and insert it to self.

        Parameters
        ----------
        name : str
            Name of the desired Constant. It will be prepended with this dictionary's
            prefix.
        value : array-like
            Initial value of constant.

        Returns
        -------
        :py:class:`.Constant`
            The created or retrieved :py:class:`.Constant`.
        """
        name = self.prefix + name
        param = self._get_impl(name)
        if param is None:
            if value is None:
                raise KeyError("No constant named '{}'. Please specify value " \
                               "if you want to create a new constant.".format(
                                   name))
            param = Constant(name, value)
            self._params[name] = param
        elif value is not None:
            assert isinstance(param, Constant), \
                "Parameter '{}' already exists but it is not a constant.".format(
                    name)
            if isinstance(value, ndarray.NDArray):
                value = value.asnumpy()
            assert param.shape == value.shape and \
                (param.value.asnumpy() == value).all(), \
                "Constant '{}' already exists but it's value doesn't match new " \
                "value".format(name)
        return param

    def update(self, other):
        """Copies all Parameters in ``other`` to self."""
        for k, v in other.items():
            if k in self._params:
                assert self._params[k] is v, \
                    "Cannot update self with other because they have different " \
                    "Parameters with the same name '%s'"%k

        for k, v in other.items():
            self._params[k] = v

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        """Initializes all Parameters managed by this dictionary to be used for :py:class:`NDArray`
        API. It has no effect when using :py:class:`Symbol` API.

        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when :py:meth:`Parameter.init` is ``None``.
            Otherwise, :py:meth:`Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        verbose : bool, default False
            Whether to verbosely print out details on initialization.
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
        if verbose:
            init.set_verbosity(verbose=verbose)
        for _, v in self.items():
            v.initialize(None, ctx, init, force_reinit=force_reinit)

    def zero_grad(self):
        """Sets all Parameters' gradient buffer to 0."""
        # collect gradient arrays for each ctx
        arrays = defaultdict(list)
        for p in self.values():
            if p.grad_req == 'null' or p._grad is None:
                continue
            for g in p.list_grad():
                if g.stype == 'row_sparse':
                    ndarray.zeros_like(g, out=g)
                else:
                    arrays[g.ctx].append(g)

        if len(arrays) == 0:
            return

        if is_np_array():
            for arr in arrays.values():
                for ele in arr:
                    ele[()] = 0
        else:
            for arr in arrays.values():
                ndarray.reset_arrays(*arr, num_arrays=len(arr))

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts.

        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        for i in self.values():
            i.reset_ctx(ctx)

    def list_ctx(self):
        """Returns a list of all the contexts on which the underlying Parameters
        are initialized."""
        s = set()
        for i in self.values():
            s.update(i.list_ctx())
        return list(s)

    def setattr(self, name, value):
        """Set an attribute to a new value for all Parameters.

        For example, set grad_req to null if you don't need gradient w.r.t a
        model's Parameters::

            model.collect_params().setattr('grad_req', 'null')

        or change the learning rate multiplier::

            model.collect_params().setattr('lr_mult', 0.5)

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : valid type for attribute name
            The new value for the attribute.
        """
        for i in self.values():
            setattr(i, name, value)

    def save(self, filename, strip_prefix=''):
        """Save parameters to file.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        strip_prefix : str, default ''
            Strip prefix from parameter names before saving.
        """
        arg_dict = {}
        for param in self.values():
            weight = param._reduce()
            if not param.name.startswith(strip_prefix):
                raise ValueError(
                    "Prefix '%s' is to be striped before saving, but Parameter's "
                    "name '%s' does not start with '%s'. "
                    "this may be due to your Block shares parameters from other "
                    "Blocks or you forgot to use 'with name_scope()' when creating "
                    "child blocks. For more info on naming, please see "
                    "https://mxnet.io/api/python/docs/tutorials/packages/gluon/blocks/naming.html"%(
                        strip_prefix, param.name, strip_prefix))
            arg_dict[param.name[len(strip_prefix):]] = weight
        ndarray.save(filename, arg_dict)

    def load(self, filename, ctx=None, allow_missing=False,
             ignore_extra=False, restore_prefix='', cast_dtype=False,
             dtype_source="current"):
        """Load parameters from file.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this ParameterDict.
        restore_prefix : str, default ''
            prepend prefix to names of stored parameters before loading.
        cast_dtype : bool, default False
            Cast the data type of the parameter
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        """
        if restore_prefix:
            for name in self.keys():
                assert name.startswith(restore_prefix), \
                    "restore_prefix is '%s' but Parameters name '%s' does not start " \
                    "with '%s'. For more info on naming, please see " \
                    "https://mxnet.io/api/python/docs/tutorials/packages/gluon/blocks/naming.html"%(
                        restore_prefix, name, restore_prefix)
        ndarray_load = ndarray.load(filename)
        self.load_dict(ndarray_load, ctx, allow_missing,
                       ignore_extra, restore_prefix, filename, cast_dtype, dtype_source)

    def load_dict(self, param_dict, ctx=None, allow_missing=False,
                  ignore_extra=False, restore_prefix='', filename=None, cast_dtype=False,
                  dtype_source="current"):
        """Load parameters from dict

        Parameters
        ----------
        param_dict : dict
            Dictionary containing model parameters, preprended with arg: and aux: names
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represented in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this ParameterDict.
        restore_prefix : str, default ''
            prepend prefix to names of stored parameters before loading
        filename : str, default None
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any
        """
        lprefix = len(restore_prefix)
        loaded = [(k[4:] if k.startswith('arg:') or k.startswith('aux:') else k, v) \
                  for k, v in param_dict.items()] if isinstance(param_dict, dict) else param_dict
        arg_dict = {restore_prefix+k: v for k, v in loaded}
        error_str = "file: %s" % (filename) if filename else "param_dict"
        if not allow_missing:
            for name in self.keys():
                assert name in arg_dict, \
                    "Parameter '%s' is missing in %s, which contains parameters: %s. " \
                    "Please make sure source and target networks have the same prefix." \
                    "For more info on naming, please see " \
                    "https://mxnet.io/api/python/docs/tutorials/packages/gluon/blocks/naming.html"%(
                        name[lprefix:], error_str, _brief_print_list(arg_dict.keys()))
        for name in arg_dict:
            if name not in self._params:
                assert ignore_extra, \
                    "Parameter '%s' loaded from %s is not present in ParameterDict, " \
                    "choices are: %s. Set ignore_extra to True to ignore. " \
                    "Please make sure source and target networks have the same prefix." \
                    "For more info on naming, please see " \
                    "https://mxnet.io/api/python/docs/tutorials/packages/gluon/blocks/naming.html"%(
                        name[lprefix:], error_str, _brief_print_list(self._params.keys()))
                continue
            self[name]._load_init(arg_dict[name], ctx, cast_dtype=cast_dtype,
                                  dtype_source=dtype_source)
