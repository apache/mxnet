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
# pylint: disable= arguments-differ, too-many-lines
"""Base container class for all neural network models."""
__all__ = ['Block', 'HybridBlock', 'SymbolBlock']

import copy
import warnings
import re
from collections import OrderedDict

from .. import symbol, ndarray, initializer
from ..symbol import Symbol
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list


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
            return self
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


def _flatten(args, inout_str):
    if isinstance(args, NDArray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)

    assert isinstance(args, (list, tuple)), \
        "HybridBlock %s must be (nested) list of Symbol or NDArray, " \
        "but got %s of type %s"%(inout_str, str(args), str(type(args)))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i, inout_str)
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
    will collect their Parameters recursively. You can also manually register
    child blocks with :py:meth:`register_child`.

    Parameters
    ----------
    prefix : str
        Prefix acts like a name space. All children blocks created in parent block's
        :py:meth:`name_scope` will have parent block's prefix in their name.
        Please refer to
        `naming tutorial <http://mxnet.incubator.apache.org/tutorials/basic/naming.html>`_
        for more info on prefix and naming.
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
        self._children = OrderedDict()
        self._reg_params = {}

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self.__dict__.items() if isinstance(block, Block)])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __setattr__(self, name, value):
        """Registers parameters."""

        if hasattr(self, name):
            existing = getattr(self, name)
            if isinstance(existing, (Parameter, Block)) and not isinstance(value, type(existing)):
                raise TypeError('Changing attribute type for {name} from {type1} to {type2}' \
                                'is not allowed.'.format(
                                    name=name, type1=type(existing), type2=type(value)))

        if isinstance(value, Block):
            self.register_child(value, name)
        elif isinstance(value, Parameter):
            assert name not in self._reg_params, \
                "Overriding Parameter attribute %s is not allowed. " \
                "If you want to share parameters between blocks, please set " \
                "'params' at Block construction instead."
            self._reg_params[name] = value

        super(Block, self).__setattr__(name, value)

    def _check_container_with_block(self):
        def _find_block_in_container(data):
            # Find whether a nested container structure contains Blocks
            if isinstance(data, (list, tuple)):
                for ele in data:
                    if _find_block_in_container(ele):
                        return True
                return False
            elif isinstance(data, dict):
                for _, v in data.items():
                    if _find_block_in_container(v):
                        return True
                return False
            elif isinstance(data, Block):
                return True
            else:
                return False
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple, dict)) and not (k.startswith('__') or k == '_children'):
                if _find_block_in_container(v):
                    warnings.warn('"{name}" is a container with Blocks. '
                                  'Note that Blocks inside the list, tuple or dict will not be '
                                  'registered automatically. Make sure to register them using '
                                  'register_child() or switching to '
                                  'nn.Sequential/nn.HybridSequential instead. '
                                  .format(name=self.__class__.__name__ + "." + k), stacklevel=3)

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

        Please refer to
        `naming tutorial <http://mxnet.incubator.apache.org/tutorials/basic/naming.html>`_
        for more info on prefix and naming.
        """
        return self._scope

    @property
    def params(self):
        """Returns this :py:class:`Block`'s parameter dictionary (does not include its
        children's parameters)."""
        return self._params

    def collect_params(self, select=None):
        """Returns a :py:class:`ParameterDict` containing this :py:class:`Block` and all of its
        children's Parameters(default), also can returns the select :py:class:`ParameterDict`
        which match some given regular expressions.

        For example, collect the specified parameters in ['conv1_weight', 'conv1_bias', 'fc_weight',
        'fc_bias']::

            model.collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')

        or collect all parameters whose names end with 'weight' or 'bias', this can be done
        using regular expressions::

            model.collect_params('.*weight|.*bias')

        Parameters
        ----------
        select : str
            regular expressions

        Returns
        -------
        The selected :py:class:`ParameterDict`
        """
        # We need to check here because blocks inside containers are not supported.
        self._check_container_with_block()
        ret = ParameterDict(self._params.prefix)
        if not select:
            ret.update(self.params)
        else:
            pattern = re.compile(select)
            ret.update({name:value for name, value in self.params.items() if pattern.match(name)})
        for cld in self._children.values():
            ret.update(cld.collect_params(select=select))
        return ret

    def _collect_params_with_prefix(self, prefix=''):
        if prefix:
            prefix += '.'
        ret = {prefix + key : val for key, val in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    def save_parameters(self, filename):
        """Save parameters to file.

        Saved parameters can only be loaded with `load_parameters`. Note that this method
        only saves parameters, not model structure. If you want to save model structures,
        please use :py:meth:`HybridBlock.export`.

        Parameters
        ----------
        filename : str
            Path to file.

        References
        ----------
        `Saving and Loading Gluon Models
        <https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html>`_
        """
        params = self._collect_params_with_prefix()
        arg_dict = {key : val._reduce() for key, val in params.items()}
        ndarray.save(filename, arg_dict)

    def save_params(self, filename):
        """[Deprecated] Please use save_parameters. Note that if you want to load
        from SymbolBlock later, please use export instead.

        Save parameters to file.

        filename : str
            Path to file.
        """
        warnings.warn("save_params is deprecated. Please use save_parameters. "
                      "Note that if you want to load from SymbolBlock later, please "
                      "use export instead. For details, see "
                      "https://mxnet.incubator.apache.org/tutorials/gluon/save_lo"
                      "ad_params.html")
        try:
            self.collect_params().save(filename, strip_prefix=self.prefix)
        except ValueError as e:
            raise ValueError('%s\nsave_params is deprecated. Using ' \
                              'save_parameters may resolve this error.'%e.message)

    def load_parameters(self, filename, ctx=None, allow_missing=False,
                        ignore_extra=False):
        """Load parameters from file previously saved by `save_parameters`.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.

        References
        ----------
        `Saving and Loading Gluon Models
        <https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html>`_
        """
        loaded = ndarray.load(filename)
        params = self._collect_params_with_prefix()
        if not loaded and not params:
            return

        if not any('.' in i for i in loaded.keys()):
            # legacy loading
            del loaded
            self.collect_params().load(
                filename, ctx, allow_missing, ignore_extra, self.prefix)
            return

        if not allow_missing:
            for name in params.keys():
                assert name in loaded, \
                    "Parameter '%s' is missing in file '%s', which contains parameters: %s. " \
                    "Set allow_missing=True to ignore missing parameters."%(
                        name, filename, _brief_print_list(loaded.keys()))
        for name in loaded:
            if not ignore_extra and name not in params:
                raise ValueError(
                    "Parameter '%s' loaded from file '%s' is not present in ParameterDict, " \
                    "which contains parameters %s. Set ignore_extra=True to ignore. "%(
                        name, filename, _brief_print_list(self._params.keys())))
            params[name]._load_init(loaded[name], ctx)


    def load_params(self, filename, ctx=None, allow_missing=False,
                    ignore_extra=False):
        """[Deprecated] Please use load_parameters.

        Load parameters from file.

        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.
        """
        warnings.warn("load_params is deprecated. Please use load_parameters.")
        self.load_parameters(filename, ctx, allow_missing, ignore_extra)

    def register_child(self, block, name=None):
        """Registers block as a child of self. :py:class:`Block` s assigned to self as
        attributes will be registered automatically."""
        if name is None:
            name = str(len(self._children))
        self._children[name] = block

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        """Initializes :py:class:`Parameter` s of this :py:class:`Block` and its children.
        Equivalent to ``block.collect_params().initialize(...)``

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
        self.collect_params().initialize(init, ctx, verbose, force_reinit)

    def hybridize(self, active=True, **kwargs):
        """Activates or deactivates :py:class:`HybridBlock` s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        **kwargs : string
            Additional flags for hybridized operator.
        """
        for cld in self._children.values():
            cld.hybridize(active, **kwargs)

    def cast(self, dtype):
        """Cast this Block to use another data type.

        Parameters
        ----------
        dtype : str or numpy.dtype
            The new data type.
        """
        for child in self._children.values():
            child.cast(dtype)
        for _, param in self.params.items():
            param.cast(dtype)

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

    `HybridBlock` is similar to `Block`, with a few differences::

        import mxnet as mx
        from mxnet.gluon import HybridBlock, nn

        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
                # use name_scope to give child Blocks appropriate names.
                with self.name_scope():
                    self.dense0 = nn.Dense(20)
                    self.dense1 = nn.Dense(20)

            def hybrid_forward(self, F, x):
                x = F.relu(self.dense0(x))
                return F.relu(self.dense1(x))

        model = Model()
        model.initialize(ctx=mx.cpu(0))
        model.hybridize()
        model(mx.nd.zeros((10, 10), ctx=mx.cpu(0)))

    Forward computation in :py:class:`HybridBlock` must be static to work with :py:class:`Symbol` s,
    i.e. you cannot call :py:meth:`NDArray.asnumpy`, :py:attr:`NDArray.shape`,
    :py:attr:`NDArray.dtype`, `NDArray` indexing (`x[i]`) etc on tensors.
    Also, you cannot use branching or loop logic that bases on non-constant
    expressions like random numbers or intermediate results, since they change
    the graph structure for each iteration.

    Before activating with :py:meth:`hybridize()`, :py:class:`HybridBlock` works just like normal
    :py:class:`Block`. After activation, :py:class:`HybridBlock` will create a symbolic graph
    representing the forward computation and cache it. On subsequent forwards,
    the cached graph will be used instead of :py:meth:`hybrid_forward`.

    Please see references for detailed tutorial.

    References
    ----------
        `Hybrid - Faster training and easy deployment
        <http://mxnet.io/tutorials/gluon/hybrid.html>`_
    """
    def __init__(self, prefix=None, params=None):
        super(HybridBlock, self).__init__(prefix=prefix, params=params)
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = None
        self._out_format = None
        self._in_format = None
        self._active = False
        self._flags = {}

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(HybridBlock, self).__setattr__(name, value)
        if isinstance(value, HybridBlock):
            self._clear_cached_op()

    def _get_graph(self, *args):
        if not self._cached_graph:
            args, self._in_format = _flatten(args, "input")
            if len(args) > 1:
                inputs = [symbol.var('data%d'%i) for i in range(len(args))]
            else:
                inputs = [symbol.var('data')]
            grouped_inputs = _regroup(inputs, self._in_format)[0]

            params = {i: j.var() for i, j in self._reg_params.items()}
            with self.name_scope():
                out = self.hybrid_forward(symbol, *grouped_inputs, **params)  # pylint: disable=no-value-for-parameter
            out, self._out_format = _flatten(out, "output")

            self._cached_graph = inputs, symbol.Group(out)

        return self._cached_graph

    def _build_cache(self, *args):
        inputs, out = self._get_graph(*args)
        input_idx = {var.name: i for i, var in enumerate(inputs)}
        self._cached_op = ndarray.CachedOp(out, self._flags)
        params = dict(self.collect_params().items())

        # verify graph inputs
        expected_inputs = set(out.list_inputs())
        for name in expected_inputs:
            assert name in params or name in input_idx, \
                "Unknown input to HybridBlock: %s"%name
        for name, i in input_idx.items():
            if name not in expected_inputs:
                warnings.warn("The %d-th input to HybridBlock is not used by any "
                              "computation. Is this intended?"%i, stacklevel=4)
        for name in params:
            if name not in expected_inputs:
                warnings.warn("Parameter %s is not used by any computation. "
                              "Is this intended?"%name, stacklevel=4)

        self._cached_op_args = [(False, params[name]) if name in params
                                else (True, input_idx[name])
                                for name in out.list_inputs()]

    def _finish_deferred_init(self, hybrid, *args):
        try:
            self.infer_shape(*args)
        except Exception as e:
            error_msg = "Deferred initialization failed because shape"\
                        " cannot be inferred \n {}".format(e)
            raise ValueError(error_msg)

        if hybrid:
            for is_arg, i in self._cached_op_args:
                if not is_arg:
                    i._finish_deferred_init()
        else:
            for _, i in self.params.items():
                i._finish_deferred_init()

    def _call_cached_op(self, *args):
        if self._cached_op is None:
            self._build_cache(*args)

        args, fmt = _flatten(args, "input")
        assert fmt == self._in_format, "Invalid input format"
        cargs = [args[i] if is_arg else i.data()
                 for is_arg, i in self._cached_op_args]
        out = self._cached_op(*cargs)
        if isinstance(out, NDArray):
            out = [out]
        return _regroup(out, self._out_format)[0]

    def _clear_cached_op(self):
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = None

    def register_child(self, block, name=None):
        if not isinstance(block, HybridBlock):
            raise ValueError(
                "Children of HybridBlock must also be HybridBlock, " \
                "but %s has type %s. If you are using Sequential, " \
                "please try HybridSequential instead."%(
                    str(block), str(type(block))))
        super(HybridBlock, self).register_child(block, name)
        self._clear_cached_op()

    def hybridize(self, active=True, **kwargs):
        self._active = active
        self._flags = kwargs.items()
        self._clear_cached_op()
        super(HybridBlock, self).hybridize(active, **kwargs)

    def cast(self, dtype):
        self._clear_cached_op()
        super(HybridBlock, self).cast(dtype)

    def _infer_attrs(self, infer_fn, attr, *args):
        """Generic infer attributes."""
        inputs, out = self._get_graph(*args)
        args, _ = _flatten(args, "input")
        with warnings.catch_warnings(record=True) as w:
            arg_attrs, _, aux_attrs = getattr(out, infer_fn)(
                **{i.name: getattr(j, attr) for i, j in zip(inputs, args)})
            if arg_attrs is None:
                raise ValueError(w[0].message)
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_attrs)}
        sdict.update({name : attr for name, attr in \
             zip(out.list_auxiliary_states(), aux_attrs)})
        for i in self.collect_params().values():
            setattr(i, attr, sdict[i.name])

    def infer_shape(self, *args):
        """Infers shape of Parameters from inputs."""
        self._infer_attrs('infer_shape', 'shape', *args)

    def infer_type(self, *args):
        """Infers data type of Parameters from inputs."""
        self._infer_attrs('infer_type', 'dtype', *args)

    def export(self, path, epoch=0):
        """Export HybridBlock to json format that can be loaded by
        `SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface.

        .. note:: When there are only one input, it will have name `data`. When there
                  Are more than one inputs, they will be named as `data0`, `data1`, etc.

        Parameters
        ----------
        path : str
            Path to save model. Two files `path-symbol.json` and `path-xxxx.params`
            will be created, where xxxx is the 4 digits epoch number.
        epoch : int
            Epoch number of saved model.
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
        ndarray.save('%s-%04d.params'%(path, epoch), arg_dict)

    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        if isinstance(x, NDArray):
            with x.context as ctx:
                try:
                    if self._active:
                        return self._call_cached_op(x, *args)
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                except DeferredInitializationError:
                    self._finish_deferred_init(self._active, x, *args)

                if self._active:
                    return self._call_cached_op(x, *args)
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
    as feature extractors. For example, you may want to extract the output
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
    @staticmethod
    def imports(symbol_file, input_names, param_file=None, ctx=None):
        """Import model previously saved by `HybridBlock.export` or
        `Module.save_checkpoint` as a SymbolBlock for use in Gluon.

        Parameters
        ----------
        symbol_file : str
            Path to symbol file.
        input_names : list of str
            List of input variable names
        param_file : str, optional
            Path to parameter file.
        ctx : Context, default None
            The context to initialize SymbolBlock on.

        Returns
        -------
        SymbolBlock
            SymbolBlock loaded from symbol and parameter files.

        Examples
        --------
        >>> net1 = gluon.model_zoo.vision.resnet18_v1(
        ...     prefix='resnet', pretrained=True)
        >>> net1.hybridize()
        >>> x = mx.nd.random.normal(shape=(1, 3, 32, 32))
        >>> out1 = net1(x)
        >>> net1.export('net1', epoch=1)
        >>>
        >>> net2 = gluon.SymbolBlock.imports(
        ...     'net1-symbol.json', ['data'], 'net1-0001.params')
        >>> out2 = net2(x)
        """
        sym = symbol.load(symbol_file)
        if isinstance(input_names, str):
            input_names = [input_names]
        inputs = [symbol.var(i) for i in input_names]
        ret = SymbolBlock(sym, inputs)
        if param_file is not None:
            ret.collect_params().load(param_file, ctx=ctx)
        return ret


    def __init__(self, outputs, inputs, params=None):
        super(SymbolBlock, self).__init__(prefix=None, params=None)
        self._prefix = ''
        self._params = ParameterDict('', params)
        if isinstance(inputs, symbol.Symbol) and len(inputs.list_outputs()) == 1:
            inputs = [inputs]
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            outputs = outputs[0]

        syms, self._in_format = _flatten(inputs, "input")
        out, self._out_format = _flatten(outputs, "output")
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
                try:
                    return self._call_cached_op(x, *args)
                except DeferredInitializationError:
                    self._finish_deferred_init(True, x, *args)

                return self._call_cached_op(x, *args)

        assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
        args, in_fmt = _flatten([x] + list(args), "input")
        assert in_fmt == self._in_format, "Invalid input format"
        ret = copy.copy(self._cached_graph[1])
        ret._compose(**{k.name: v for k, v in zip(self._cached_graph[0], args)})
        return _regroup(list(ret), self._out_format)[0]

    def _clear_cached_op(self):
        tmp = self._cached_graph
        super(SymbolBlock, self)._clear_cached_op()
        self._cached_graph = tmp

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError
