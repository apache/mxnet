# coding: utf-8
# pylint: disable= arguments-differ
"""Neural network layers."""

from .. import symbol, ndarray
from ..symbol import Symbol
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError


class _LayerScope(object):
    """Scope for collecting sub-layers."""
    _current = None

    def __init__(self, layer):
        self._layer = layer
        self._counter = {}
        self._old_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Create prefix and params for new layer."""
        current = _LayerScope._current
        if current is None:
            if prefix is None:
                prefix = _name.NameManager.current.get(None, hint) + '_'
            if params is None:
                params = ParameterDict(prefix)
            else:
                params = ParameterDict(params.prefix, params)
            return prefix, params

        if prefix is None:
            count = current._counter.get(hint, 0)
            prefix = '%s%d_'%(hint, count)
            current._counter[hint] = count + 1
        if params is None:
            parent = current._layer.params
            params = ParameterDict(parent.prefix+prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        return current._layer.prefix+prefix, params

    def __enter__(self):
        self._old_scope = _LayerScope._current
        _LayerScope._current = self
        return self

    def __exit__(self, ptype, value, trace):
        _LayerScope._current = self._old_scope


def _flatten(args):
    if isinstance(args, NDArray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)

    assert isinstance(args, (list, tuple)), \
        "HybridLayer input must be (nested) list of Symbol or NDArray, " \
        "but got %s of type %s"%(str(args), str(type(args)))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i)
        flat.extend(arg)
        fmts.append(fmt)
    return flat, fmts


def _regroup(args, fmt):
    if isinstance(fmt, int):
        if fmt == 0:
            return args[0], args[1:]
        return args[:fmt], args[fmt:]

    assert isinstance(args, (list, tuple)), \
        "HybridLayer output must be (nested) list of Symbol or NDArray, " \
        "but got %s of type %s"%(str(args), str(type(args)))
    ret = []
    for i in fmt:
        res, args = _regroup(args, i)
        ret.append(res)
    return ret, args


class Layer(object):
    """Base class for all neural network layers and models.

    Your models should subclass this class.

    Layers can also contain other Layers, allowing you to nest them in a tree
    structure. You can assign sublayers as regular attributes::
        from mxnet import nn
        class Net(nn.Layer):
            def __init__(self, **kwargs):
                super(Net, self).__init__(**kwargs)
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)

            def forward(self, x):
                x = self.dense0(x)
                return self.dense1(x)

    Sublayers assigned this way will be registered and will have their status changed
    too when you call .train() etc.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. It will be prepended to the name of all Symbols and
        Parameters created by this layer. Prefix should be unique within one network
        to prevent name collisions.
    params : ParameterDict or None
        ParameterDict for sharing weights with the new Layer. For example,
        if you want `dense2` to share `dense1`'s weights, you can do::
            dense1 = nn.Dense(20, in_units=10, prefix='dense1_')
            dense2 = nn.Dense(20, in_units=10, prefix='dense2_',
                              params=dense1.all_params())

    Layer supports forwarding with both `Symbol` and `NDArray`."""
    def __init__(self, prefix=None, params=None):
        self._prefix, self._params = _LayerScope.create(prefix, params, self._alias())
        self._scope = _LayerScope(self)
        self._children = []

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(Layer, self).__setattr__(name, value)
        if isinstance(value, Layer):
            self.register_child(value)

    def _alias(self):
        return self.__class__.__name__.lower()

    @property
    def params(self):
        """Returns this Layer's parameter dictionary (does not include its
        children's parameters)."""
        return self._params

    def all_params(self):
        """Returns a ParameterDict containing this Layer and all of its children's
        Parameters."""
        ret = ParameterDict(self._params.prefix)
        ret.update(self.params)
        for cld in self._children:
            ret.update(cld.all_params())
        return ret

    @property
    def prefix(self):
        """Prefix of this Layer."""
        return self._prefix

    @property
    def name(self):
        if self.prefix.endswith('_'):
            return self.prefix[:-1]
        return self.prefix

    def name_scope(self):
        """Returns a name space object managing sublayer and parameter
        names. Should be used by `with` statement
        """
        return self._scope

    def register_child(self, layer):
        """Register layer as sublayer of self. Layers assigned to
        self as attributes will be registered automatically."""
        self._children.append(layer)

    def hybridize(self, active=True):
        """Activate HybridLayers recursively. Has no effect on
        non-hybrid children."""
        for cld in self._children:
            cld.hybridize(active)

    def __call__(self, *args):
        """Call forward."""
        return self.forward(*args)

    def forward(self, *args):
        """Override to implement forward computation using NDArray.

        Parameters
        ----------
        *args : list of NDArray
            Input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class HybridLayer(Layer):
    """HybridLayer supports forwarding with both Symbol and NDArray.

    Forward computation in HybridLayer must be static to work with Symbols,
    i.e. you cannot call `.asnumpy()`, `.shape`, `.dtype`, etc on inputs.
    When forwarding after `hybridize()` is called, HybridLayer will
    create a graph representing the forward computation and cache it.
    On subsequent forward the cached graph will be used instead of calling
    `hybrid_forward`.
    """
    def __init__(self, prefix=None, params=None):
        super(HybridLayer, self).__init__(prefix=prefix, params=params)
        self._reg_params = {}
        self._cached_graph = ()
        self._cached_op = None
        self._cached_params = None
        self._out_format = None
        self._in_format = None
        self._active = False

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(HybridLayer, self).__setattr__(name, value)
        if isinstance(value, Parameter):
            assert name not in self._reg_params or \
                not isinstance(self._reg_params[name], Parameter), \
                "Overriding Parameter attribute %s is not allowed. " \
                "Please pass in Parameters by specifying `params` at " \
                "Layer construction instead."
            self._reg_params[name] = value

    def register_child(self, layer):
        if not isinstance(layer, HybridLayer):
            if isinstance(layer, Sequential):
                raise ValueError(
                    "Children of HybridLayer must also be HybridLayer. " \
                    "Please use HSequential instead of Sequantial.")
            raise ValueError(
                "Children of HybridLayer must also be HybridLayer, " \
                "but %s has type %s."%(str(layer), str(type(layer))))
        super(HybridLayer, self).register_child(layer)

    def hybridize(self, active=True):
        super(HybridLayer, self).hybridize(active)
        self._active = active

    def _get_graph(self, *args):
        if self._cached_graph:
            return self._cached_graph

        args, self._in_format = _flatten(args)
        syms = [symbol.var(str(i)) for i in range(len(args))]
        sym_args = _regroup(syms, self._in_format)[0]

        params = {i: j.var() for i, j in self._reg_params.items()}
        out = self.hybrid_forward(symbol, *sym_args, **params)  # pylint: disable=no-value-for-parameter
        out, self._out_format = _flatten(out)

        self._cached_graph = syms, symbol.Group(out)
        return self._cached_graph

    def infer_shape(self, *args):
        """Infer shape of Parameters from inputs."""
        syms, out = self._get_graph(*args)
        args, _, = _flatten(args)
        arg_shapes, _, aux_shapes = out.infer_shape(
            **{i.name: j.shape for i, j in zip(syms, args)})
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_shapes)}
        sdict.update({name : shape for name, shape in \
                      zip(out.list_auxiliary_states(), aux_shapes)})
        for i in self.all_params().values():
            i.shape = sdict[i.name]

    def _build_cache(self, *args):
        self.infer_shape(*args)
        for i in self.all_params().values():
            i._finish_deferred_init()

        _, out = self._get_graph(*args)
        self._cached_op = ndarray.CachedOp(out)
        params = dict(self.all_params().items())
        self._cached_params = [params.get(name, None) for name in out.list_inputs()]
        self._in_idx = [(i, int(name)) for i, name in enumerate(out.list_inputs())
                        if name not in params]

    def _call_cached_op(self, *args):
        args, fmt = _flatten(args)
        assert fmt == self._in_format, "Invalid input format"
        cargs = [i.data() if i else None for i in self._cached_params]
        for i, j in self._in_idx:
            cargs[i] = args[j]
        out = self._cached_op(*cargs)
        if isinstance(out, NDArray):
            out = [out]
        return _regroup(out, self._out_format)[0]

    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        NDArray or Symbol."""
        if isinstance(x, NDArray):
            if self._active and self._cached_op is None:
                self._build_cache(x, *args)

            with x.context as ctx:
                if self._active:
                    return self._call_cached_op(x, *args)
                try:
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                except DeferredInitializationError:
                    self.infer_shape(x, *args)
                    for i in self.all_params().values():
                        i._finish_deferred_init()
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                return self.hybrid_forward(ndarray, x, *args, **params)
        else:
            assert isinstance(x, Symbol), \
                "Layer requires the first argument to forward be either " \
                "Symbol or NDArray, but got %s"%type(x)
            params = {i: j.var() for i, j in self._reg_params.items()}
            return self.hybrid_forward(symbol, x, *args, **params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Override to construct symbolic graph for this Layer.

        Parameters
        ----------
        x : Symbol
            The first input Symbol.
        *args : list of Symbol
            Additional input Symbols.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError
