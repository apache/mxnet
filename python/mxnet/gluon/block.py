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
# pylint: disable= arguments-differ
"""Base container class for all neural network models."""
__all__ = ['Block', 'HybridBlock', 'SymbolBlock']

import copy

from .. import symbol, ndarray, initializer
from ..symbol import Symbol
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent


class _BlockScope(object):
    """Scope for collecting child `Block` s."""
    _current = None

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

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
        if self._block._empty_prefix:
            return
        self._old_scope = _BlockScope._current
        _BlockScope._current = self
        self._name_scope = _name.Prefix(self._block.prefix)
        self._name_scope.__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
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

    :py:class:`Block` can be nested recursively in a tree structure. You can create and
    assign child :py:class:`Block` as regular attributes::

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


    Child :py:class:`Block` assigned this way will be registered and :py:meth:`collect_params`
    will collect their Parameters recursively.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. It will be prepended to the names of all
        Parameters and child :py:class:`Block` s in this :py:class:`Block` 's
        :py:meth:`name_scope` .
        Prefix should be unique within one model to prevent name collisions.
    params : ParameterDict or None
        :py:class:`ParameterDict` for sharing weights with the new :py:class:`Block`. For example,
        if you want ``dense1`` to share ``dense0``'s weights, you can do::

            dense0 = nn.Dense(20)
            dense1 = nn.Dense(20, params=dense0.collect_params())
    """
    def __init__(self, prefix=None, params=None):
        self._empty_prefix = prefix == ''
        self._prefix, self._params = _BlockScope.create(prefix, params, self._alias())
        self._name = self._prefix[:-1] if self._prefix.endswith('_') else self._prefix
        self._scope = _BlockScope(self)
        self._children = []

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self.__dict__.items() if isinstance(block, Block)])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

    def __setattr__(self, name, value):
        """Registers parameters."""

        if hasattr(self, name):
            existing = getattr(self, name)
            if isinstance(existing, (Parameter, Block)) and not isinstance(value, type(existing)):
                raise TypeError('Changing attribute type for {name} from {type1} to {type2}' \
                                'is not allowed.'.format(name=name,
                                                         type1=type(existing),
                                                         type2=type(value)))
            if isinstance(existing, Block):
                for i, c in enumerate(self._children):
                    if c is existing:
                        self._children[i] = value
        else:
            if isinstance(value, Block):
                self.register_child(value)

        super(Block, self).__setattr__(name, value)

    def _alias(self):
        return self.__class__.__name__.lower()

    @property
    def prefix(self):
        """Prefix of this :py:class:`Block`."""
        return self._prefix

    @property
    def name(self):
        """Name of this :py:class:`Block`, without '_' in the end."""
        return self._name

    def name_scope(self):
        """Returns a name space object managing a child :py:class:`Block` and parameter
        names. Should be used within a ``with`` statement::

            with self.name_scope():
                self.dense = nn.Dense(20)
        """
        return self._scope

    @property
    def params(self):
        """Returns this :py:class:`Block`'s parameter dictionary (does not include its
        children's parameters)."""
        return self._params

    def collect_params(self):
        """Returns a :py:class:`ParameterDict` containing this :py:class:`Block` and all of its
        children's Parameters."""
        ret = ParameterDict(self._params.prefix)
        ret.update(self.params)
        for cld in self._children:
            ret.update(cld.collect_params())
        return ret

    def save_params(self, filename):
        """Save parameters to file.

        filename : str
            Path to file.
        """
        self.collect_params().save(filename, strip_prefix=self.prefix)

    def load_params(self, filename, ctx, allow_missing=False,
                    ignore_extra=False):
        """Load parameters from file.

        filename : str
            Path to parameter file.
        ctx : Context or list of Context
            Context(s) initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
        """
        self.collect_params().load(filename, ctx, allow_missing, ignore_extra,
                                   self.prefix)


    def register_child(self, block):
        """Registers block as a child of self. :py:class:`Block` s assigned to self as
        attributes will be registered automatically."""
        self._children.append(block)

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False):
        """Initializes :py:class:`Parameter` s of this :py:class:`Block` and its children.

        Equivalent to ``block.collect_params().initialize(...)``
        """
        self.collect_params().initialize(init, ctx, verbose)

    def hybridize(self, active=True):
        """Activates or deactivates :py:class:`HybridBlock` s recursively. Has no effect on
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
        """Overrides to implement forward computation using :py:class:`NDArray`. Only
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

    Forward computation in :py:class:`HybridBlock` must be static to work with :py:class:`Symbol` s,
    i.e. you cannot call :py:meth:`NDArray.asnumpy`, :py:attr:`NDArray.shape`,
    :py:attr:`NDArray.dtype`, etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with :py:meth:`hybridize()`, :py:class:`HybridBlock` works just like normal
    :py:class:`Block`. After activation, :py:class:`HybridBlock` will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of :py:meth:`hybrid_forward`.

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

    def _get_graph(self, *args):
        if not self._cached_graph:
            args, self._in_format = _flatten(args)
            if len(args) > 1:
                inputs = [symbol.var('data%d'%i) for i in range(len(args))]
            else:
                inputs = [symbol.var('data')]
            grouped_inputs = _regroup(inputs, self._in_format)[0]

            params = {i: j.var() for i, j in self._reg_params.items()}
            with self.name_scope():
                out = self.hybrid_forward(symbol, *grouped_inputs, **params)  # pylint: disable=no-value-for-parameter
            out, self._out_format = _flatten(out)

            self._cached_graph = inputs, symbol.Group(out)

        return self._cached_graph

    def _build_cache(self, *args):
        inputs, out = self._get_graph(*args)
        self._cached_op = ndarray.CachedOp(out)

        params = dict(self.collect_params().items())
        self._cached_params = [params.get(name, None) for name in out.list_inputs()]
        assert len(params) + len(self._cached_graph[0]) == len(out.list_inputs()), \
            "Wrong number of inputs."

        name2pos = {var.name: i for i, var in enumerate(inputs)}
        self._in_idx = [(i, name2pos[name]) for i, name in enumerate(out.list_inputs())
                        if name not in params]

    def _call_cached_op(self, *args):
        if self._cached_op is None:
            self._build_cache(*args)

        try:
            cargs = [i.data() if i else None for i in self._cached_params]
        except DeferredInitializationError:
            self.infer_shape(*args)
            for i in self._cached_params:
                if i is not None:
                    i._finish_deferred_init()
            cargs = [i.data() if i else None for i in self._cached_params]

        args, fmt = _flatten(args)
        assert fmt == self._in_format, "Invalid input format"
        for i, j in self._in_idx:
            cargs[i] = args[j]
        out = self._cached_op(*cargs)
        if isinstance(out, NDArray):
            out = [out]
        return _regroup(out, self._out_format)[0]

    def _clear_cached_op(self):
        self._cached_graph = ()
        self._cached_op = None

    def register_child(self, block):
        if not isinstance(block, HybridBlock):
            raise ValueError(
                "Children of HybridBlock must also be HybridBlock, " \
                "but %s has type %s. If you are using Sequential, " \
                "please try HybridSequential instead"%(
                    str(block), str(type(block))))
        super(HybridBlock, self).register_child(block)
        self._clear_cached_op()

    def hybridize(self, active=True):
        self._active = active
        super(HybridBlock, self).hybridize(active)

    def infer_shape(self, *args):
        """Infers shape of Parameters from inputs."""
        inputs, out = self._get_graph(*args)
        args, _ = _flatten(args)
        arg_shapes, _, aux_shapes = out.infer_shape(
            **{i.name: j.shape for i, j in zip(inputs, args)})
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_shapes)}
        sdict.update({name : shape for name, shape in \
                      zip(out.list_auxiliary_states(), aux_shapes)})
        for i in self.collect_params().values():
            i.shape = sdict[i.name]

    def export(self, path):
        """Export HybridBlock to json format that can be loaded by `mxnet.mod.Module`
        or the C++ interface.

        .. note:: When there are only one input, it will have name `data`. When there
                  Are more than one inputs, they will be named as `data0`, `data1`, etc.

        Parameters
        ----------
        path : str
            Path to save model. Two files `path-symbol.json` and `path-0000.params`
            will be created.
        """
        if not self._cached_graph:
            raise RuntimeError(
                "Please first call block.hybridize() and then run forward with "
                "this block at least once before calling export.")
        sym = self._cached_graph[1]
        sym.save('%s-symbol.json'%path)

        arg_names = set(sym.list_arguments())
        aux_names = set(sym.list_auxiliary_states())
        arg_dict = {}
        for name, param in self.collect_params().items():
            if name in arg_names:
                arg_dict['arg:%s'%name] = param._reduce()
            else:
                assert name in aux_names
                arg_dict['aux:%s'%name] = param._reduce()
        ndarray.save('%s-0000.params'%path, arg_dict)

    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        if isinstance(x, NDArray):
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

        assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
        params = {i: j.var() for i, j in self._reg_params.items()}
        with self.name_scope():
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


class SymbolBlock(HybridBlock):
    """Construct block from symbol. This is useful for using pre-trained models
    as feature extractors. For example, you may want to extract get the output
    from fc2 layer in AlexNet.

    Parameters
    ----------
    outputs : Symbol or list of Symbol
        The desired output for SymbolBlock.
    inputs : Symbol or list of Symbol
        The Variables in output's argument that should be used as inputs.
    params : ParameterDict
        Parameter dictionary for arguments and auxililary states of outputs
        that are not inputs.

    Examples
    --------
    >>> # To extract the feature from fc1 and fc2 layers of AlexNet:
    >>> alexnet = gluon.model_zoo.vision.alexnet(pretrained=True, ctx=mx.cpu(),
                                                 prefix='model_')
    >>> inputs = mx.sym.var('data')
    >>> out = alexnet(inputs)
    >>> internals = out.get_internals()
    >>> print(internals.list_outputs())
    ['data', ..., 'model_dense0_relu_fwd_output', ..., 'model_dense1_relu_fwd_output', ...]
    >>> outputs = [internals['model_dense0_relu_fwd_output'],
                   internals['model_dense1_relu_fwd_output']]
    >>> # Create SymbolBlock that shares parameters with alexnet
    >>> feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())
    >>> x = mx.nd.random.normal(shape=(16, 3, 224, 224))
    >>> print(feat_model(x))
    """
    def __init__(self, outputs, inputs, params=None):
        super(SymbolBlock, self).__init__(prefix=None, params=None)
        self._prefix = ''
        self._params = ParameterDict('', params)
        if isinstance(inputs, symbol.Symbol) and len(inputs.list_outputs()) == 1:
            inputs = [inputs]
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            outputs = outputs[0]

        syms, self._in_format = _flatten(inputs)
        out, self._out_format = _flatten(outputs)
        out = symbol.Group(out)

        input_names = set()
        for i in syms:
            assert len(i.get_internals().list_outputs()) == 1, \
                "Input symbols must be variable, but %s is an output of operators"%str(i)
            input_names.add(i.name)

        for i in out.list_arguments():
            if i not in input_names:
                self.params.get(i, allow_deferred_init=True)

        for i in out.list_auxiliary_states():
            if i not in input_names:
                self.params.get(i, grad_req='null', allow_deferred_init=True)

        self._cached_graph = syms, out
        self._build_cache()

    def forward(self, x, *args):
        if isinstance(x, NDArray):
            with x.context:
                return self._call_cached_op(x, *args)

        assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
        args, in_fmt = _flatten([x] + list(args))
        assert in_fmt == self._in_format, "Invalid input format"
        ret = copy.copy(self._cached_graph[1])
        ret._compose(**{k.name: v for k, v in zip(self._cached_graph[0], args)})
        return _regroup(list(ret), self._out_format)[0]

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError
