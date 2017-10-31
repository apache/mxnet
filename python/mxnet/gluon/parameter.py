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
# pylint: disable=
"""Neural network parameter."""
__all__ = ['DeferredInitializationError', 'Parameter', 'ParameterDict',
           'tensor_types']

from collections import OrderedDict
import warnings
import numpy as np


from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context
from .. import autograd
from .utils import _indent

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
    shape : tuple of int, default None
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
                 differentiable=True):
        self._var = None
        self._data = None
        self._grad = None
        self._ctx_list = None
        self._ctx_map = None
        self._deferred_init = ()
        self._differentiable = differentiable
        self._allow_deferred_init = allow_deferred_init
        self._grad_req = None
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init

    def __repr__(self):
        s = 'Parameter {name} (shape={shape}, dtype={dtype})'
        return s.format(**self.__dict__)

    @property
    def grad_req(self):
        return self._grad_req

    @grad_req.setter
    def grad_req(self, req):
        assert req in ['write', 'add', 'null'], \
            "grad_req must be one of write, add, or null, but got %s"%req
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

    def _check_and_get(self, arr_list, ctx):
        if arr_list is not None:
            if ctx is list:
                return arr_list
            if ctx is None:
                if len(arr_list) == 1:
                    return arr_list[0]
                else:
                    ctx = context.current_context()
            idx = self._ctx_map[ctx.device_typeid][ctx.device_id]
            if idx is not None:
                return arr_list[idx]
            raise RuntimeError(
                "Parameter %s was not initialized on context %s. "
                "It was only initialized on %s."%(
                    self.name, str(ctx), str(self._ctx_list)))
        if self._deferred_init:
            raise DeferredInitializationError(
                "Parameter %s has not been initialized yet because initialization was " \
                "deferred. Actual initialization happens during the first forward pass. " \
                "Please pass one batch of data through the network before accessing Parameters. " \
                "You can also avoid deferred initialization by specifying in_units, " \
                "num_features, etc., for network layers."%(self.name))
        raise RuntimeError(
            "Parameter %s has not been initialized. Note that " \
            "you should initialize parameters and create Trainer " \
            "with Block.collect_params() instead of Block.params " \
            "because the later does not include Parameters of " \
            "nested child Blocks"%(self.name))

    def _load_init(self, data, ctx):
        """(Re)initializes by loading from data."""
        if self.shape:
            for i, j in zip(self.shape, data.shape):
                assert i == 0 or i == j, \
                    "Failed loading Parameter %s from saved params: " \
                    "shape incompatible expacted %s vs saved %s"%(
                        self.name, str(self.shape), str(data.shape))
        if self.dtype:
            assert np.dtype(self.dtype).type == data.dtype, \
                "Failed loading Parameter %s from saved params: " \
                "dtype incompatible expacted %s vs saved %s"%(
                    self.name, str(self.dtype), str(data.dtype))
        if isinstance(ctx, Context):
            ctx = [ctx]
        if self._data is None:
            if self._deferred_init:
                assert set(ctx) == set(self._deferred_init[1]), \
                    "Failed to load Parameter %s on %s because it was " \
                    "previous initialized on %s."%(
                        self.name, str(ctx), str(self.list_ctx()))
            self._init_impl(data, ctx)
        else:
            assert set(ctx) == set(self.list_ctx()), \
                "Failed to load Parameter %s on %s because it was " \
                "previous initialized on %s."%(
                    self.name, str(ctx), str(self.list_ctx()))
            self.set_data(data)
        self._deferred_init = ()

    def _finish_deferred_init(self):
        """Finishes deferred initialization."""
        if not self._deferred_init:
            return
        init, ctx, default_init = self._deferred_init
        self._deferred_init = ()
        assert self.shape is not None and np.prod(self.shape) > 0, \
            "Cannot initialize Parameter %s because it has " \
            "invalid shape: %s. Please specify in_units, " \
            "in_channels, etc for `Block`s."%(
                self.name, str(self.shape))

        with autograd.pause():
            data = ndarray.zeros(shape=self.shape, dtype=self.dtype,
                                 ctx=context.cpu())
            initializer.create(default_init)(
                initializer.InitDesc(self.name, {'__init__': init}), data)

            self._init_impl(data, ctx)

    def _init_impl(self, data, ctx_list):
        """Sets data and grad."""
        self._ctx_list = list(ctx_list)
        self._ctx_map = []
        for i, ctx in enumerate(self._ctx_list):
            while len(self._ctx_map) <= ctx.device_typeid:
                self._ctx_map.append([])
            dev_list = self._ctx_map[ctx.device_typeid]
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

        self._grad = [ndarray.zeros_like(i) for i in self._data]
        autograd.mark_variables(self.list_data(), self.list_grad(), self.grad_req)

    def _reduce(self):
        """Reduce data from multiple context."""
        block = self.list_data()
        data = ndarray.add_n(*(w.copyto(context.cpu()) for w in block)) / len(block)
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
            warnings.warn("Parameter %s is already initialized, ignoring. " \
                          "Set force_reinit=True to re-initialize."%self.name)
            return
        self._data = self._grad = None

        if ctx is None:
            ctx = [context.current_context()]
        if isinstance(ctx, Context):
            ctx = [ctx]
        if init is None:
            init = default_init if self.init is None else self.init
        if not self.shape or np.prod(self.shape) <= 0:
            if self._allow_deferred_init:
                self._deferred_init = (init, ctx, default_init)
                return
            raise ValueError("Cannot initialize Parameter %s because it has " \
                             "invalid shape: %s."%(self.name, str(self.shape)))

        self._deferred_init = (init, ctx, default_init)
        self._finish_deferred_init()

    def reset_ctx(self, ctx):
        """Re-assign Parameter to other contexts.

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
            init, _, default_init = self._deferred_init
            self._deferred_init = (init, ctx, default_init)
        else:
            raise ValueError("Cannot reset context for Parameter %s because it "
                             "has not been initialized."%self.name)


    def set_data(self, data):
        """Sets this parameter's value on all contexts to data."""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        for arr in self.list_data():
            arr[:] = data

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before.

        Parameters
        ----------
        ctx : Context
            Desired context.

        Returns
        -------
        NDArray on ctx
        """
        return self._check_and_get(self._data, ctx)

    def list_data(self):
        """Returns copies of this parameter on all contexts, in the same order
        as creation."""
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
                "Cannot get gradient array for Parameter %s " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, ctx)

    def list_grad(self):
        """Returns gradient buffers on all contexts, in the same order
        as :py:meth:`values`."""
        if self._data is not None and self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter %s " \
                "because grad_req='null'"%(self.name))
        return self._check_and_get(self._grad, list)

    def list_ctx(self):
        """Returns a list of contexts this parameter is initialized on."""
        if self._data is None:
            if self._deferred_init:
                return self._deferred_init[1]
            raise RuntimeError("Parameter %s has not been initialized"%self.name)
        return self._ctx_list

    def zero_grad(self):
        """Sets gradient buffer on all contexts to 0. No action is taken if
        parameter is uninitialized or doesn't require gradient."""
        if self._grad is None:
            return
        for i in self._grad:
            i[:] = 0

    def var(self):
        """Returns a symbol representing this parameter."""
        if self._var is None:
            self._var = symbol.var(self.name, shape=self.shape, dtype=self.dtype,
                                   lr_mult=self.lr_mult, wd_mult=self.wd_mult,
                                   init=self.init)
        return self._var


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
        if param is None:
            param = Parameter(name, **kwargs)
            self._params[name] = param
        else:
            for k, v in kwargs.items():
                if hasattr(param, k) and getattr(param, k) is not None:
                    assert v is None or v == getattr(param, k), \
                        "Cannot retrieve Parameter %s because desired attribute " \
                        "does not match with stored for attribute %s: " \
                        "desired %s vs stored %s."%(
                            name, k, str(v), str(getattr(param, k)))
                else:
                    setattr(param, k, v)
        return param

    def update(self, other):
        """Copies all Parameters in ``other`` to self."""
        for k, v in other.items():
            if k in self._params:
                assert self._params[k] is v, \
                    "Cannot update self with other because they have different " \
                    "Parameters with the same name %s"%k
            else:
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
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
        if verbose:
            init.set_verbosity(verbose=verbose)
        for _, v in self.items():
            v.initialize(None, ctx, init, force_reinit=force_reinit)

    def zero_grad(self):
        """Sets all Parameters' gradient buffer to 0."""
        for i in self.values():
            i.zero_grad()

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts.

        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        for i in self.values():
            i.reset_ctx(ctx)

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
                    "Prefix %s is to be striped before saving, but Parameter " \
                    "%s does not start with %s. If you are using Block.save_params, " \
                    "This may be due to your Block shares parameters from other " \
                    "Blocks or you forgot to use ``with name_scope()`` during init. " \
                    "Consider switching to Block.collect_params.save and " \
                    "Block.collect_params.load instead."%(
                        strip_prefix, param.name, strip_prefix))
            arg_dict[param.name[len(strip_prefix):]] = weight
        ndarray.save(filename, arg_dict)

    def load(self, filename, ctx, allow_missing=False,
             ignore_extra=False, restore_prefix=''):
        """Load parameters from file.

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
        """
        if restore_prefix:
            for name in self.keys():
                assert name.startswith(restore_prefix), \
                    "restore_prefix is %s but Parameters name %s does not start " \
                    "with %s"%(restore_prefix, name, restore_prefix)
        lprefix = len(restore_prefix)
        loaded = [(k[4:] if k.startswith('arg:') or k.startswith('aux:') else k, v) \
                  for k, v in ndarray.load(filename).items()]
        arg_dict = {restore_prefix+k: v for k, v in loaded}
        if not allow_missing:
            for name in self.keys():
                assert name in arg_dict, \
                    "Parameter %s is missing in file %s"%(name[lprefix:], filename)
        for name in arg_dict:
            if name not in self._params:
                assert ignore_extra, \
                    "Parameter %s loaded from file %s is not present in ParameterDict"%(
                        name[lprefix:], filename)
                continue
            self[name]._load_init(arg_dict[name], ctx)
