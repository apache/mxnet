# coding: utf-8
# pylint: disable= arguments-differ
"""Base container class for all neural network models."""

from .. import symbol, ndarray, initializer
from ..symbol import Symbol
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent


class _BlockScope(object):
    """Scope for collecting child `Block`s."""
    _current = None

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Creates prefix and params for new `Block`."""
        current = _BlockScope._current
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
            parent = current._block.params
            params = ParameterDict(parent.prefix+prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        return current._block.prefix+prefix, params

    def __enter__(self):
        self._old_scope = _BlockScope._current
        _BlockScope._current = self
        return self

    def __exit__(self, ptype, value, trace):
        _BlockScope._current = self._old_scope


def _flatten(args):
    if isinstance(args, NDArray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)

    assert isinstance(args, (list, tuple)), \
        "HybridBlock input must be (nested) list of Symbol or NDArray, " \
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
        "HybridBlock output must be (nested) list of Symbol or NDArray, " \
        "but got %s of type %s"%(str(args), str(type(args)))
    ret = []
    for i in fmt:
        res, args = _regroup(args, i)
        ret.append(res)
    return ret, args


class Block(object):
    """Base class for all neural network layers and models. Your models should
    subclass this class.

    `Block` can be nested recursively in a tree structure. You can create and
    assign child `Block` as regular attributes::

        from mxnet.gluon import Block, nn
        from mxnet import ndarray as F

        class Model(Block):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                # use name_scope to give child Blocks appropriate names.
                # It also allows sharing Parameters between Blocks recursively.
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)

            def forward(self, x):
                x = F.relu(self.dense0(x))
                return F.relu(self.dense1(x))

        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model(F.zeros((10, 10), ctx=mx.cpu(0)))


    Child `Block` assigned this way will be registered and `collect_params`
    will collect their Parameters recursively.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. It will be prepended to the name of all
        Parameters and child `Block`s in this `Block`'s `name_scope`. Prefix
        should be unique within one model to prevent name collisions.
    params : ParameterDict or None
        `ParameterDict` for sharing weights with the new `Block`. For example,
        if you want `dense1` to share `dense0`'s weights, you can do::

            dense0 = nn.Dense(20)
            dense1 = nn.Dense(20, params=dense0.collect_params())
    """
    def __init__(self, prefix=None, params=None):
        self._prefix, self._params = _BlockScope.create(prefix, params, self._alias())
        self._scope = _BlockScope(self)
        self._children = []

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(Block, self).__setattr__(name, value)
        if isinstance(value, Block):
            self.register_child(value)

    def _alias(self):
        return self.__class__.__name__.lower()

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self.__dict__.items() if isinstance(block, Block)])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

    @property
    def params(self):
        """Returns this `Block`'s parameter dictionary (does not include its
        children's parameters)."""
        return self._params

    def collect_params(self):
        """Returns a `ParameterDict` containing this `Block` and all of its
        children's Parameters."""
        ret = ParameterDict(self._params.prefix)
        ret.update(self.params)
        for cld in self._children:
            ret.update(cld.collect_params())
        return ret

    @property
    def prefix(self):
        """Prefix of this `Block`."""
        return self._prefix

    @property
    def name(self):
        """Name of this `Block`, without '_' in the end."""
        if self.prefix.endswith('_'):
            return self.prefix[:-1]
        return self.prefix

    def name_scope(self):
        """Returns a name space object managing a child `Block` and parameter
        names. Should be used within a `with` statement::

            with self.name_scope():
                self.dense = nn.Dense(20)
        """
        return self._scope

    def register_child(self, block):
        """Registers block as a child of self. `Block`s assigned to self as
        attributes will be registered automatically."""
        self._children.append(block)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False):
        """Initializes `Parameter`s of this `Block` and its children.

        Equivalent to `block.collect_params().initialize(...)`
        """
        self.collect_params().initialize(init, ctx, verbose)

    def hybridize(self, active=True):
        """Activates or deactivates `HybridBlock`s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        """
        for cld in self._children:
            cld.hybridize(active)

    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        return self.forward(*args)

    def forward(self, *args):
        """Overrides to implement forward computation using `NDArray`. Only
        accepts positional arguments.

        Parameters
        ----------
        *args : list of NDArray
            Input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class HybridBlock(Block):
    """`HybridBlock` supports forwarding with both Symbol and NDArray.

    Forward computation in `HybridBlock` must be static to work with `Symbol`s,
    i.e. you cannot call `.asnumpy()`, `.shape`, `.dtype`, etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with `hybridize()`, `HybridBlock` works just like normal
    `Block`. After activation, `HybridBlock` will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of `hybrid_forward`.

    Refer `Hybrid tutorial <http://mxnet.io/tutorials/gluon/hybrid.html>`_ to see
    the end-to-end usage.
    """
    def __init__(self, prefix=None, params=None):
        super(HybridBlock, self).__init__(prefix=prefix, params=params)
        self._reg_params = {}
        self._cached_graph = ()
        self._cached_op = None
        self._cached_params = None
        self._out_format = None
        self._in_format = None
        self._active = False

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(HybridBlock, self).__setattr__(name, value)
        if isinstance(value, Parameter):
            assert name not in self._reg_params or \
                not isinstance(self._reg_params[name], Parameter), \
                "Overriding Parameter attribute %s is not allowed. " \
                "Please pass in Parameters by specifying `params` at " \
                "Block construction instead."
            self._reg_params[name] = value

    def register_child(self, block):
        if not isinstance(block, HybridBlock):
            raise ValueError(
                "Children of HybridBlock must also be HybridBlock, " \
                "but %s has type %s. If you are using Sequential, " \
                "please try HybridSequential instead"%(
                    str(block), str(type(block))))
        super(HybridBlock, self).register_child(block)

    def hybridize(self, active=True):
        self._active = active
        super(HybridBlock, self).hybridize(active)

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
        """Infers shape of Parameters from inputs."""
        syms, out = self._get_graph(*args)
        args, _, = _flatten(args)
        arg_shapes, _, aux_shapes = out.infer_shape(
            **{i.name: j.shape for i, j in zip(syms, args)})
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_shapes)}
        sdict.update({name : shape for name, shape in \
                      zip(out.list_auxiliary_states(), aux_shapes)})
        for i in self.collect_params().values():
            i.shape = sdict[i.name]

    def _build_cache(self, *args):
        self.infer_shape(*args)
        for i in self.collect_params().values():
            i._finish_deferred_init()

        _, out = self._get_graph(*args)
        self._cached_op = ndarray.CachedOp(out)
        params = dict(self.collect_params().items())
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
        `NDArray` or `Symbol`."""
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
                    for i in self.collect_params().values():
                        i._finish_deferred_init()
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                return self.hybrid_forward(ndarray, x, *args, **params)
        else:
            assert isinstance(x, Symbol), \
                "HybridBlock requires the first argument to forward be either " \
                "Symbol or NDArray, but got %s"%type(x)
            params = {i: j.var() for i, j in self._reg_params.items()}
            return self.hybrid_forward(symbol, x, *args, **params)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError
