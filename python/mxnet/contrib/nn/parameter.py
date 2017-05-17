# coding: utf-8
# pylint: disable=
"""Neural network parameter."""

from collections import OrderedDict
import numpy as np

from ...base import mx_real_t
from ... import symbol, ndarray, initializer, context
from ...context import Context
from ...contrib import autograd

# pylint: disable= invalid-name
tensor_types = (symbol.Symbol, ndarray.NDArray)
# pylint: enable= invalid-name

class Parameter(object):
    """A Container holding parameters (weights) of layers.

    `Parameter` can be used with both `Symbol` and `NDArray` API. For `Symbol` API,
    `Parameter.var()` will return a `Symbol` representing this parameter. It
    can then be used for composing networks::
        x = mx.sym.Variable('data')
        w = mx.nn.Parameter('fc_weight', init=mx.init.Xavier())
        b = mx.nn.Parameter('fc_bias', init=mx.init.Zero())
        out = mx.sym.FullyConnected(x, w.var(), b.var(), num_hidden=64)

    For `NDArray` API, `Parameter` must be initialized with `Parameter.init`. It
    will then hold a copy of the the parameter on each `Context`. If `grad_req` is
    not `null`, it will also hold a gradient array on each `Context`::
        ctx = mx.gpu(0)
        x = mx.nd.zeros((16, 100), ctx=ctx)
        w = mx.nn.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())
        b = mx.nn.Parameter('fc_bias', shape(64,), init=mx.init.Zero())
        w.initialize(ctx=ctx)
        b.initialize(ctx=ctx)
        out = mx.nd.FullyConnected(x, w.value(ctx), b.value(ctx), num_hidden=64)

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
        - 'null' means gradient is not reqested for this parameter. gradient arrays
          will not be allocated.
    shape : tuple of int, default None
        Shape of this parameter. By default shape is not specified. Parameter with
        unknown shaped can be used for `Symbol` API, but `init` will throw an error
        when using `NDArray` API.
    dtype : numpy.dtype or str, default 'float32'
        Data type of this parameter. For example, numpy.float32 or 'float32'.
    lr_mult : float, default 1.0
        Learning rate multiplier. Learning rate will be multiplied by lr_mult
        when updating this parameter with optimizer.
    wd_mult : float, default 1.0
        Weight decay multiplier (L2 regulerizer coefficient). Works similarly to lr_mult.
    init : Initializer, default None
        Initializer of this parameter. Will use the global initializer by default.
    """
    def __init__(self, name, grad_req='write', shape=None, dtype=mx_real_t,
                 lr_mult=1.0, wd_mult=1.0, init=None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.lr_mult = lr_mult
        self.wd_mult = wd_mult
        self.grad_req = grad_req
        self.init = init
        self._var = None
        self._data = None
        self._grad = None

    def initialize(self, init=None, ctx=None, default_init=initializer.Xavier()):
        """Intialize parameter and gradient arrays. Only used for `NDArray` API.

        init : Initializer
            The initializer to use. Overrides `Parameter.init` and default_init.
        ctx : Context or list of Context, defaults to `context.current_context()`.
            Initialize Parameter on given context. If ctx is a list of Context, a
            copy will be made for each context.

            .. note:: Copies are independent arrays. User is responsible for keeping
            their values consistent when updating. Normally nn.Optim does this for you.
        default_init : Initializer
            Default initializer is used when both `init` and `Parameter.init` are None.
        """
        if ctx is None:
            ctx = [context.current_context()]
        if isinstance(ctx, Context):
            ctx = [ctx]

        assert np.prod(self.shape) > 0, \
            "Cannot initialize Parameter %s because it has invalid shape: %s. " \
            "Please specify in_units, in_filters, etc for Layers"%(
                self.name, str(self.shape))
        data = ndarray.zeros(shape=self.shape, dtype=self.dtype, ctx=ctx[0])
        if init is None:
            init = self.init
        initializer.create(default_init)(
            initializer.InitDesc(self.name, {'__init__': init}),
            data)

        self._data = OrderedDict()
        self._data[ctx[0]] = data
        for i in ctx[1:]:
            self._data[i] = data.copyto(i)

        if self.grad_req == 'null':
            self._grad = None
            return

        self._grad = OrderedDict()
        for i in ctx:
            self._grad[i] = ndarray.zeros_like(self._data[i])

        autograd.mark_variables(self.list_data(), self.list_grad(), self.grad_req)

    def set(self, data):
        """Set this parameter's value on all contexts to data."""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        for arr in self.list_data():
            arr[:] = data

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must be on this context
        before.

        Parameters
        ----------
        ctx : Context
            Desired context.

        Returns
        -------
        NDArray on ctx
        """
        if ctx is None:
            ctx = Context.current_context()
        assert self._data is not None, \
            "Cannot get NDArray value for Parameter %s " \
            "because it hasn't been initialized!"%(self.name)
        assert ctx in self._data, \
            "Cannot get NDArray value for Parameter %s on context %s " \
            "because it was not initialized on %s"%(self.name, str(ctx), str(ctx))
        return self._data[ctx]

    def list_data(self):
        """Returns copies of this parameter on all contexts, in the same order
        as creation."""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        return self._data.values()

    def grad(self, ctx=None):
        """Returns a gradient buffer for this parameter on one context.

        Parameters
        ----------
        ctx : Context
            Desired context.
        """
        if ctx is None:
            ctx = Context.current_context()
        assert self._grad is not None, \
            "Cannot get gradient array for Parameter %s " \
            "because it hasn't been initialized or grad_req='null'"%(self.name)
        assert ctx in self._grad, \
            "Cannot get gradient array for Parameter %s on context %s " \
            "because it was not initialized on %s"%(self.name, str(ctx), str(ctx))
        return self._grad[ctx]

    def list_grad(self):
        """Returns gradient buffers on all contexts, in the same order
        as `values`."""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        assert self._data is not None, \
            "Parameter %s does not have gradients because grad_req='null'"%self.name
        return self._grad.values()

    def list_ctx(self):
        """Returns a list of contexts this parameter is initialized on"""
        assert self._data is not None, \
            "Parameter %s has not been initialized"%self.name
        return self._data.keys()

    def zero_grad(self):
        """Set gradient buffer on all contexts to 0. No action is taken if
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
    """
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def __getitem__(self, key):
        return self._params[key]

    def items(self):
        return self._params.items()

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    @property
    def prefix(self):
        """Prefix of this dict. It will be prepended to Parameters' name created
        with `get`"""
        return self._prefix

    def get(self, name, **kwargs):
        """Create or retrieve a Parameter with name `self.prefix+name`. Key-word
        arguments will be passed to Parameter's contructor.

        Parameter
        ---------
        name : str
            name of the desired Parameter. It will be prepended with this dictionary's
            prefix.
        **kwargs : dict
            The rest of key-word arguments for the created Parameter.

        Returns
        -------
        Parameter
            The created or retrieved Parameter.
        """
        name = self.prefix + name
        if name not in self._params:
            self._params[name] = Parameter(name, **kwargs)
        else:
            param = self._params[name]
            for k, v in kwargs.items():
                if hasattr(param, k):
                    assert v is None or v == getattr(param, k), \
                        "Parameter attribute %s mismatch: stored %s vs desired %s"%(
                            k, str(getattr(param, k)), str(v))
                else:
                    setattr(param, k, v)
        return self._params[name]

    def subdict(self, suffix):
        """Create a sub-dictionary that shares parameters with this dictionary.
        The sub-dictionary's prefix is self.prefix + suffix.

        Example::
            >>> params1 = ParameterDict('net_')
            >>> params2 = params1.subdict('conv1_')
            >>> params2.prefix
            'net_conv1_'

        Parameters
        ----------
        suffix : str
            Suffix of the created child dictionary

        Returns
        -------
        ParameterDict with self.prefix + suffix as prefix.
        """
        ret = ParameterDict(self.prefix + suffix)
        self.merge(ret)
        return ret

    def merge(self, other):
        """Merge this dictionary with another dictionary. The two dictionaries
        will manage the same set of Parameters but keep their individual prefix.

        Example::
            >>> params1 = ParameterDict('net1_')
            >>> params2 = ParameterDict('net2_')
            >>> params1.merge(params2)
            >>> params2.get('w')
            >>> print params1.keys()
            ['net2_w']
        """
        params = self._params
        if params is other._params:
            return
        for k, v in other.items():
            assert k not in params or params[k] is v, \
                "Cannot merge ParameterDicts with prefix %s and %s " \
                "because they contain different versions of the same " \
                "Parameter named %s"%(self.prefix, other.prefix, k)
            params[k] = v
        other._params = params

    def initialize(self, init=initializer.Xavier(), ctx=None):
        """Intialize all Parameters manage by this dictionary to be used for `NDArray`
        API. Has no effect when using `Symbol` API.

        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when `Parameter.init` is None.
            Otherwise `Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keep a copy of Parameters on one or many context(s).
        """
        for _, v in self.items():
            v.initialize(None, ctx, init)

    def zero_grad(self):
        """Set all Parameters' gradient buffer to 0."""
        for i in self.values():
            i.zero_grad()

    def save(self, filename):
        arg_dict = {}
        for param in self.values():
            block = param.list_data()
            weight = sum(w.copyto(context.cpu()) for w in block) / len(block)
            arg_dict[param.name] = weight
        ndarray.save(filename, arg_dict)

    def load(self, filename, allow_missing=False, ignore_extra=False):
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
            self[name].set(arg_dict[name])
