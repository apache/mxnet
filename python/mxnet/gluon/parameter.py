# coding: utf-8
# pylint: disable=
"""Neural network parameter."""

from collections import OrderedDict
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
    """A Container holding parameters (weights) of `Block`s.

    `Parameter` holds a copy of the the parameter on each `Context` after
    it is initialized with `Parameter.initialize(...)`. If `grad_req` is
    not `null`, it will also hold a gradient array on each `Context`::

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

        - 'write' means everytime gradient is written to grad `NDArray`.
        - 'add' means everytime gradient is added to the grad `NDArray`. You need
          to manually call `zero_grad()` to clear the gradient buffer before each
          iteration when using this option.
        - 'null' means gradient is not requested for this parameter. gradient arrays
          will not be allocated.
    shape : tuple of int, default None
        Shape of this parameter. By default shape is not specified. Parameter with
        unknown shape can be used for `Symbol` API, but `init` will throw an error
        when using `NDArray` API.
    dtype : numpy.dtype or str, default 'float32'
        Data type of this parameter. For example, numpy.float32 or 'float32'.
    lr_mult : float, default 1.0
        Learning rate multiplier. Learning rate will be multiplied by lr_mult
        when updating this parameter with optimizer.
    wd_mult : float, default 1.0
        Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.
    init : Initializer, default None
        Initializer of this parameter. Will use the global initializer by default.

    """
    def __init__(self, name, grad_req='write', shape=None, dtype=mx_real_t,
                 lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init
        self.allow_deferred_init = allow_deferred_init
        self._var = None
        self._data = None
        self._grad = None
        self._defered_init = ()

    def __repr__(self):
        s = 'Parameter {name} (shape={shape}, dtype={dtype})'
        return s.format(**self.__dict__)

    def initialize(self, init=None, ctx=None, default_init=initializer.Uniform()):
        """Initializes parameter and gradient arrays. Only used for `NDArray` API.

        Parameters
        ----------
        init : Initializer
            The initializer to use. Overrides `Parameter.init` and default_init.
        ctx : Context or list of Context, defaults to `context.current_context()`.
            Initialize Parameter on given context. If ctx is a list of Context, a
            copy will be made for each context.

            .. note:: Copies are independent arrays. User is responsible for keeping
            their values consistent when updating. Normally `gluon.Trainer` does this for you.
        default_init : Initializer
            Default initializer is used when both `init` and `Parameter.init` are `None`.

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
        if ctx is None:
            ctx = [context.current_context()]
        if isinstance(ctx, Context):
            ctx = [ctx]
        if init is None:
            init = default_init if self.init is None else self.init
        if not self.shape or np.prod(self.shape) <= 0:
            if self.allow_deferred_init:
                self._defered_init = (init, ctx, default_init)
                return
            raise ValueError("Cannot initialize Parameter %s because it has " \
                             "invalid shape: %s."%(self.name, str(self.shape)))

        self._defered_init = (init, ctx, default_init)
        self._finish_deferred_init()

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
            if self._defered_init:
                assert set(ctx) == set(self._defered_init[1]), \
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
        self._defered_init = ()

    def _finish_deferred_init(self):
        """Finishes deferred initialization."""
        if not self._defered_init:
            return
        init, ctx, default_init = self._defered_init
        self._defered_init = ()
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

    def _init_impl(self, data, ctx):
        """Sets data and grad."""
        self._data = OrderedDict()
        for i in ctx:
            self._data[i] = data.copyto(i)

        if self.grad_req == 'null':
            self._grad = None
            return

        self._grad = OrderedDict()
        for i in ctx:
            self._grad[i] = ndarray.zeros_like(self._data[i])

        autograd.mark_variables(self.list_data(), self.list_grad(), self.grad_req)

    def set_data(self, data):
        """Sets this parameter's value on all contexts to data."""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        for arr in self.list_data():
            arr[:] = data

    def _check_initialized(self, ctx=None):
        if self._data is not None:
            if ctx is not None and ctx not in self._data:
                raise RuntimeError(
                    "Parameter %s was not initialized on context %s. "
                    "It was only initialized on %s."%(
                        self.name, str(ctx), str(self.list_ctx())))
            return
        if self._defered_init:
            raise DeferredInitializationError
        raise RuntimeError(
            "Parameter %s has not been initialized. Note that " \
            "you should initialize parameters and create Trainer " \
            "with Block.collect_params() instead of Block.params " \
            "because the later does not include Parameters of " \
            "nested child Blocks"%(self.name))

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
        if ctx is None:
            list_ctx = self.list_ctx()
            if len(list_ctx) == 1:
                ctx = list_ctx[0]
            else:
                ctx = context.current_context()
        self._check_initialized(ctx)
        return self._data[ctx]

    def list_data(self):
        """Returns copies of this parameter on all contexts, in the same order
        as creation."""
        self._check_initialized()
        return list(self._data.values())

    def grad(self, ctx=None):
        """Returns a gradient buffer for this parameter on one context.

        Parameters
        ----------
        ctx : Context
            Desired context.
        """
        if ctx is None:
            list_ctx = self.list_ctx()
            if len(list_ctx) == 1:
                ctx = list_ctx[0]
            else:
                ctx = context.current_context()
        self._check_initialized(ctx)
        if self._grad is None:
            raise RuntimeError(
                "Cannot get gradient array for Parameter %s " \
                "because grad_req='null'"%(self.name))
        return self._grad[ctx]

    def list_grad(self):
        """Returns gradient buffers on all contexts, in the same order
        as `values`."""
        self._check_initialized()
        assert self._grad is not None, \
            "Parameter %s does not have gradients because grad_req='null'"%self.name
        return list(self._grad.values())

    def list_ctx(self):
        """Returns a list of contexts this parameter is initialized on."""
        if self._data is None:
            if self._defered_init:
                return self._defered_init[1]
            raise RuntimeError("Parameter %s has not been initialized"%self.name)
        return list(self._data.keys())

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
    prefix : str, default ''
        The prefix to be prepended to all Parameters' name created by this dict.
    shared : ParameterDict or None
        If not `None`, when this dict's `get` method creates a new parameter, will
        first try to retrieve it from `shared` dict. Usually used for sharing
        parameters with another `Block`.
    """
    def __init__(self, prefix='', shared=None):
        self._prefix = prefix
        self._params = {}
        self._shared = shared

    def __getitem__(self, key):
        return self._params[key]

    def __repr__(self):
        s = '{name}(\n{content}\n)'
        name = self._prefix+' ' if self._prefix else ''
        return s.format(name=name,
                        content='\n'.join([_indent('  {0}'.format(v), 2)
                                           for v in self.values()]))

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    @property
    def prefix(self):
        """Prefix of this dict. It will be prepended to Parameters' name created
        with `get`."""
        return self._prefix

    def _get_impl(self, name):
        if name in self._params:
            return self._params[name]
        if self._shared is not None and name in self._shared._params:
            self._params[name] = self._shared._params[name]
            return self._shared._params[name]
        return None

    def get(self, name, **kwargs):
        """Retrieves a `Parameter` with name `self.prefix+name`. If not found,
        `get` will first try to retrieve it from `shared` dict. If still not
        found, `get` will create a new `Parameter` with key-word arguments and
        insert it to self.

        Parameters
        ----------
        name : str
            Name of the desired Parameter. It will be prepended with this dictionary's
            prefix.
        **kwargs : dict
            The rest of key-word arguments for the created `Parameter`.

        Returns
        -------
        Parameter
            The created or retrieved `Parameter`.
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
        """Copies all Parameters in `other` to self."""
        for k, v in other.items():
            if k in self._params:
                assert self._params[k] is v, \
                    "Cannot update self with other because they have different " \
                    "Parameters with the same name %s"%k
            else:
                self._params[k] = v

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False):
        """Initializes all Parameters managed by this dictionary to be used for `NDArray`
        API. It has no effect when using `Symbol` API.

        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when `Parameter.init` is `None`.
            Otherwise, `Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        """
        if verbose:
            init.set_verbosity(verbose=verbose)
        for _, v in self.items():
            v.initialize(None, ctx, init)

    def zero_grad(self):
        """Sets all Parameters' gradient buffer to 0."""
        for i in self.values():
            i.zero_grad()

    def save(self, filename):
        arg_dict = {}
        for param in self.values():
            block = param.list_data()
            weight = sum(w.copyto(context.cpu()) for w in block) / len(block)
            arg_dict[param.name] = weight
        ndarray.save(filename, arg_dict)

    def load(self, filename, ctx, allow_missing=False, ignore_extra=False):
        arg_dict = ndarray.load(filename)
        if not allow_missing:
            for name in self.keys():
                assert name in arg_dict, \
                    "Parameter %s is missing in file %s"%(name, filename)
        for name in arg_dict:
            if name not in self._params:
                assert ignore_extra, \
                    "Parameter %s loaded from file %s is not present in ParameterDict"%(
                        name, filename)
                continue
            self[name]._load_init(arg_dict[name], ctx)
