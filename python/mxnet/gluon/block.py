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
# pylint: disable= arguments-differ, too-many-lines, reimported
"""Base container class for all neural network models."""
__all__ = ['Block', 'HybridBlock', 'SymbolBlock']

import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np

from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array



class _BlockScope(object):
    """Scope for collecting child `Block` s."""
    _current = threading.local()

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Creates prefix and params for new `Block`."""
        current = getattr(_BlockScope._current, "value", None)
        if current is None:
            if prefix is None:
                if not hasattr(_name.NameManager._current, "value"):
                    _name.NameManager._current.value = _name.NameManager()
                prefix = _name.NameManager._current.value.get(None, hint) + '_'
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
        self._old_scope = getattr(_BlockScope._current, "value", None)
        _BlockScope._current.value = self
        self._name_scope = _name.Prefix(self._block.prefix)
        self._name_scope.__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
        _BlockScope._current.value = self._old_scope


def _gather_type_ctx_info(args):
    """Analyze the elements inside the nested args object and find:
        - If there exists ndarray
        - If there exists symbol
        - All contexts appearing in args

    Parameters
    ----------
    args : list or NDArray or Symbol
        Could be a nested architecture.

    Returns
    -------
    has_symbol : bool
        Whether the elements in args contains symbols
    has_ndarray : bool
        Whether the elements in args contains ndarrays
    ctx_set : set of mxnet.context.Context
        Contains all possible contexts of the inner ndarrays in args. Can be empty if there is no
        ndarray inside args.
    first_ctx : mxnet.context.Context or None
        Context of the first appeared NDArray (for backward-compatibility)
    """
    if isinstance(args, NDArray):
        return False, True, {args.ctx}, args.ctx
    elif isinstance(args, Symbol):
        return True, False, set(), None
    elif isinstance(args, (list, tuple)):
        has_symbol = False
        has_ndarray = False
        ctx_set = set()
        first_ctx = None
        for ele in args:
            ele_has_sym, ele_has_nd, ele_ctx_set, ele_first_ctx =\
                _gather_type_ctx_info(ele)
            has_symbol = has_symbol or ele_has_sym
            has_ndarray = has_ndarray or ele_has_nd
            if first_ctx is None and ele_first_ctx is not None:
                first_ctx = ele_first_ctx
            ctx_set = ctx_set | ele_ctx_set
            if has_symbol and has_ndarray:
                break
        return has_symbol, has_ndarray, ctx_set, first_ctx
    else:
        return False, False, set(), None


def _flatten(args, inout_str):
    """Parse the arguments into a flattened list + an additional format array.
    The format array stores the structure of the original arguments to help reconstruct the inputs.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    inout_str : str
        The name of the HybridBlock

    Returns
    -------
    flat : list of Symbol or NDArray
        The flatten version of the input args.
    fmts : (nested) list of ints
        Stores the format information of the original structured args.
    """
    if isinstance(args, NDArray):
        return [args], int(0)
    if isinstance(args, Symbol):
        length = len(args.list_outputs())
        length = length if length > 1 else 0
        return [args], int(length)
    if args is None:
        return [None], int(-1)

    if not isinstance(args, (list, tuple)):
        raise ValueError("When hybridized, the input of HybridBlock {}"
                         " must be (nested) list of Symbol"
                         " or NDArray, "
                         "but got {} of type {}".format(inout_str, str(args), str(type(args))))
    flat = []
    fmts = []
    for i in args:
        arg, fmt = _flatten(i, inout_str)
        flat.extend(arg)
        fmts.append(fmt)
    return flat, fmts


def _regroup(args, fmt):
    """Reconstruct the structured arguments based on the flattened version.

    Parameters
    ----------
    args : NDArray, Symbol, or (nested) list of Symbol or NDArray
        We allow None inside the args.
    fmt : (nested) list of ints
        Stores the format information of the original structured args.

    Returns
    -------
    ret : NDArray, Symbol, or (nested) list of Symbol or NDArray

    """
    def _merger(args, fmt):
        """Recursive call to merge the arguments"""
        if isinstance(fmt, int):
            if fmt < -1:
                raise ValueError("Unsupported encoded format {}.".format(fmt))
            if fmt == 0:
                return args[0], args[1:]
            if fmt == -1:
                if args[0] is not None:
                    raise ValueError('We do not support passing types that are not None'
                                     ' when the initial HybridBlock has received NoneType and'
                                     ' has been hybridized.'
                                     ' Received arg = {}, fmt = {}.'.format(args[0], fmt))
                return None, args[1:]
            else:
                return args[:fmt], args[fmt:]

        if not isinstance(args, (list, tuple)):
            raise ValueError("When hybridized, the output of HybridBlock must be (nested)"
                             " list of Symbol or NDArray, "
                             "but got {} of type {}".format(args, type(args)))
        ret = []
        for i in fmt:
            res, args = _merger(args, i)
            ret.append(res)
        return ret, args
    return _merger(args, fmt)[0]


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
        `naming tutorial </api/python/docs/tutorials/packages/gluon/blocks/naming.html>`_
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
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()

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
        children = set(self._children.values())
        def _find_unregistered_block_in_container(data):
            # Find whether a nested container structure contains Blocks
            if isinstance(data, (list, tuple)):
                for ele in data:
                    if _find_unregistered_block_in_container(ele):
                        return True
                return False
            elif isinstance(data, dict):
                for _, v in data.items():
                    if _find_unregistered_block_in_container(v):
                        return True
                return False
            elif isinstance(data, Block):
                return not data in children
            else:
                return False
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple, dict)) and not (k.startswith('__') or k == '_children'):
                if _find_unregistered_block_in_container(v):
                    warnings.warn('"{name}" is an unregistered container with Blocks. '
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
        `the naming tutorial </api/python/docs/tutorials/packages/gluon/blocks/naming.html>`_
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

    def save_parameters(self, filename, deduplicate=False):
        """Save parameters to file.

        Saved parameters can only be loaded with `load_parameters`. Note that this
        method only saves parameters, not model structure. If you want to save
        model structures, please use :py:meth:`HybridBlock.export`.

        Parameters
        ----------
        filename : str
            Path to file.
        deduplicate : bool, default False
            If True, save shared parameters only once. Otherwise, if a Block
            contains multiple sub-blocks that share parameters, each of the
            shared parameters will be separately saved for every sub-block.

        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>`_
        """
        params = self._collect_params_with_prefix()

        if deduplicate:
            # Shared parameters are stored only a single time as of MXNet 1.6.
            # Shared parameters are registered under multiple prefixes returned by
            # _collect_params_with_prefix. We select a single one and only store
            # it. In load_parameters it is sufficient for a shared parameter to
            # only set it for a single prefix.
            reverse_params = {v: k for k, v in params.items()}
            params = {v: k for k, v in reverse_params.items()}

        arg_dict = {key: val._reduce() for key, val in params.items()}
        save_fn = _mx_npx.save if is_np_array() else ndarray.save
        save_fn(filename, arg_dict)

    def save_params(self, filename):
        """[Deprecated] Please use save_parameters. Note that if you want load
        from SymbolBlock later, please use export instead.

        Save parameters to file.

        filename : str
            Path to file.
        """
        warnings.warn("save_params is deprecated. Please use save_parameters. "
                      "Note that if you want load from SymbolBlock later, please "
                      "use export instead. For details, see "
                      "https://mxnet.apache.org/tutorials/gluon/save_lo"
                      "ad_params.html")
        try:
            self.collect_params().save(filename, strip_prefix=self.prefix)
        except ValueError as e:
            raise ValueError('%s\nsave_params is deprecated. Using ' \
                              'save_parameters may resolve this error.'%e.message)

    def load_parameters(self, filename, ctx=None, allow_missing=False,
                        ignore_extra=False, cast_dtype=False, dtype_source='current'):
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
        cast_dtype : bool, default False
            Cast the data type of the NDArray loaded from the checkpoint to the dtype
            provided by the Parameter if any.
        dtype_source : str, default 'current'
            must be in {'current', 'saved'}
            Only valid if cast_dtype=True, specify the source of the dtype for casting
            the parameters
        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html>`_
        """
        if is_np_array():
            # failure may happen when loading parameters saved as NDArrays within
            # NumPy semantics. Check the failure type and recover from it if it happens.
            try:
                loaded = _mx_npx.load(filename)
            except MXNetError as e:
                err_msg = str(e)
                if 'is_np_shape' in err_msg:
                    # Loading failure due to parameters saved without numpy semantics.
                    # Temporarily disable numpy semantics and load parameters. After it's
                    # done, resume the numpy semantics. This is fine because the cases
                    # numpy ndarray covers is a superset of the legacy ndarray's.
                    with np_array(False):
                        with np_shape(False):
                            loaded_nds = ndarray.load(filename)
                    assert isinstance(loaded_nds, dict),\
                        'expecting a dict type, got {}'.format(str(type(loaded_nds)))
                    loaded = {k: loaded_nds[k].as_np_ndarray() for k in loaded_nds}
                else:
                    raise ValueError(err_msg)
        else:
            loaded = ndarray.load(filename)
        params = self._collect_params_with_prefix()
        if not loaded and not params:
            return

        if not any('.' in i for i in loaded.keys()):
            # legacy loading
            loaded = None  # This should be changed to `del loaded` when dropping Python 2
            self.collect_params().load(
                filename, ctx, allow_missing, ignore_extra, self.prefix,
                cast_dtype=cast_dtype, dtype_source=dtype_source)
            return

        if not allow_missing:
            # Shared parameters are stored only a single time as of MXNet 1.6.
            # We thus retrieve all prefixes (through _collect_params_with_prefix)
            # that a shared parameter is used with. Check that there are no
            # missing parameters that were not yet already loaded from the
            # shared version.
            params_inv = defaultdict(list)
            for k, v in params.items():
                params_inv[v].append(k)

            for name, param in params.items():
                assert any(p in loaded for p in params_inv[param]), \
                    "Parameter '%s' is missing in file '%s', which contains parameters: %s. " \
                    "Set allow_missing=True to ignore missing parameters."%(
                        name, filename, _brief_print_list(loaded.keys()))
        for name in loaded:
            if not ignore_extra and name not in params:
                raise ValueError(
                    "Parameter '%s' loaded from file '%s' is not present in ParameterDict, " \
                    "which contains parameters %s. Set ignore_extra=True to ignore. "%(
                        name, filename, _brief_print_list(self._params.keys())))
            if name in params:
                params[name]._load_init(loaded[name], ctx, cast_dtype=cast_dtype, dtype_source=dtype_source)

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

    def register_forward_pre_hook(self, hook):
        r"""Registers a forward pre-hook on the block.

        The hook function is called immediately before :func:`forward`.
        It should not modify the input or output.

        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input) -> None`.

        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_pre_hooks, hook)
        return handle

    def register_forward_hook(self, hook):
        r"""Registers a forward hook on the block.

        The hook function is called immediately after :func:`forward`.
        It should not modify the input or output.

        Parameters
        ----------
        hook : callable
            The forward hook function of form `hook(block, input, output) -> None`.

        Returns
        -------
        :class:`mxnet.gluon.utils.HookHandle`
        """
        handle = HookHandle()
        handle.attach(self._forward_hooks, hook)
        return handle

    def apply(self, fn):
        r"""Applies ``fn`` recursively to every child block as well as self.

        Parameters
        ----------
        fn : callable
            Function to be applied to each submodule, of form `fn(block)`.

        Returns
        -------
        this block
        """
        for cld in self._children.values():
            cld.apply(fn)
        fn(self)
        return self

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
        """ Please refer description of HybridBlock hybridize().
        """
        for cld in self._children.values():
            cld.hybridize(active, **kwargs)

    def save(self, prefix):
        """Save the model architecture and parameters to load again later

        Saves the model architecture as a nested dictionary where each Block
        in the model is a dictionary and its children are sub-dictionaries.

        Each Block is uniquely identified by Block class name and a unique ID.
        We save the child's name that that parent uses for it to restore later
        in order to match the saved parameters.

        Recursively traverses a Block's children in order (since its an
        OrderedDict) and uses the unique ID to denote that specific Block.
        Assumes that the model is created in an identical order every time.
        If the model is not able to be recreated deterministically do not
        use this set of APIs to save/load your model.

        For HybridBlocks, the cached_graph (Symbol & inputs) is saved if
        it has already been hybridized.

        Parameters
        ----------
        prefix : str
            The prefix to use in filenames for saving this model:
            <prefix>-model.json and <prefix>-model.params
        """
        # create empty model structure
        model = {}
        def _save_cached_graphs(blk, index, structure):
            # create new entry for this block
            mdl = {'orig_name': blk.name}
            # encode unique name based on block type and ID
            name = type(blk).__name__.lower()
            structure[name+str(index[0])] = mdl
            if isinstance(blk, HybridBlock):
                if blk._cached_graph:
                    # save in/out formats
                    mdl['in_format'] = blk._in_format
                    mdl['out_format'] = blk._out_format
                    # save cached graph & input symbols
                    syms, out = blk._cached_graph
                    mdl_syms = []
                    for sym in syms:
                        mdl_syms.append(sym.tojson())
                    mdl['inputs'] = mdl_syms
                    mdl['symbol'] = out.tojson()
                    mdl['hybridized'] = True
                else:
                    mdl['hybridized'] = False
            children = dict()
            mdl['children'] = children
            # recursively save children
            for ch_name, child in blk._children.items():
                index[0] += 1
                # save child's original name in this block's map
                children[child.name] = ch_name
                _save_cached_graphs(child, index, mdl)
        # save top-level block
        index = [0]
        _save_cached_graphs(self, index, model)
        # save model
        with open(prefix+'-model.json', 'w') as fp:
            json.dump(model, fp)
        # save params
        self.save_parameters(prefix+'-model.params')

    def load(self, prefix):
        """Load a model saved using the `save` API

        Reconfigures a model using the saved configuration. This function
        does not regenerate the model architecture. It resets the children's
        names as they were when saved in order to match the names of the
        saved parameters.

        This function assumes the Blocks in the model were created in the same
        order they were when the model was saved. This is because each Block is
        uniquely identified by Block class name and a unique ID in order (since
        its an OrderedDict) and uses the unique ID to denote that specific Block.
        Assumes that the model is created in an identical order every time.
        If the model is not able to be recreated deterministically do not
        use this set of APIs to save/load your model.

        For HybridBlocks, the cached_graph (Symbol & inputs) and settings are
        restored if it had been hybridized before saving.

        Parameters
        ----------
        prefix : str
            The prefix to use in filenames for loading this model:
            <prefix>-model.json and <prefix>-model.params
        """
        # load model json from file
        with open(prefix+'-model.json') as fp:
            model = json.load(fp)

        def _load_cached_graphs(blk, index, structure):
            # get block name
            name = type(blk).__name__.lower()
            # lookup previous encoded name based on block type and ID
            mdl = structure[name+str(index[0])]
            # rename block to what it was when saved
            blk._name = mdl['orig_name']
            if isinstance(blk, HybridBlock):
                if mdl['hybridized']:
                    # restore in/out formats
                    blk._in_format = mdl['in_format']
                    blk._out_format = mdl['out_format']
                    # get saved symbol
                    out = load_json(mdl['symbol'])
                    syms = []
                    # recreate inputs for this symbol
                    for inp in mdl['inputs']:
                        syms.append(load_json(inp))
                    # reset cached_graph and active status
                    blk._cached_graph = (syms, out)
                    blk._active = True
            # rename params with updated block name
            pnames = list(blk.params.keys())
            for p in pnames:
                param = blk.params._params[p]
                new_name = blk.name +'_'+ p[len(blk.params._prefix):]
                blk.params._params.pop(p)
                blk.params._params[new_name] = param
            # recursively reload children
            for ch_name, child in blk._children.items():
                index[0] += 1
                _load_cached_graphs(child, index, mdl)
            # current set of child names
            ch_names = list(blk._children.keys())
            # original child names
            children = mdl['children']
            # loop and remap children with original names
            for ch_name in ch_names:
                child = blk._children[ch_name]
                blk._children.pop(ch_name)
                orig_name = children[child.name]
                blk._children[orig_name] = child
        # load top-level block
        index = [0]
        _load_cached_graphs(self, index, model)
        # load params
        self.load_parameters(prefix+'-model.params')

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
        for hook in self._forward_pre_hooks.values():
            hook(self, args)

        out = self.forward(*args)

        for hook in self._forward_hooks.values():
            hook(self, args, out)
        if _mx_npx.is_np_array():
            _check_all_np_ndarrays(out)
        return out

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

    def register_op_hook(self, callback, monitor_all=False):
        """Install callback monitor.

        Parameters
        ----------
        callback : function
            Takes a string and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input and output, otherwise monitor output only.
        """
        for cld in self._children.values():
            cld.register_op_hook(callback, monitor_all)

    def summary(self, *inputs):
        """Print the summary of the model's output and parameters.

        The network must have been initialized, and must not have been hybridized.

        Parameters
        ----------
        inputs : object
            Any input that the model supports. For any tensor in the input, only
            :class:`mxnet.ndarray.NDArray` is supported.
        """
        summary = OrderedDict()
        seen = set()
        hooks = []

        def _get_shape_str(args):
            def flatten(args):
                if not isinstance(args, (list, tuple)):
                    return [args], int(0)
                flat = []
                fmts = []
                for i in args:
                    arg, fmt = flatten(i)
                    flat.extend(arg)
                    fmts.append(fmt)
                return flat, fmts

            def regroup(args, fmt):
                if isinstance(fmt, int):
                    if fmt == 0:
                        return args[0], args[1:]
                    return args[:fmt], args[fmt:]
                ret = []
                for i in fmt:
                    res, args = regroup(args, i)
                    ret.append(res)
                return ret, args

            flat_args, fmts = flatten(args)
            flat_arg_shapes = [x.shape if isinstance(x, ndarray.NDArray) else x
                               for x in flat_args]
            shapes = regroup(flat_arg_shapes, fmts)[0]
            if isinstance(shapes, list):
                shape_str = str(shapes)[1:-1]
            else:
                shape_str = str(shapes)
            return shape_str.replace('L', '')

        def _register_summary_hook(block):
            assert not isinstance(block, HybridBlock) or not block._active, \
                    '"{}" must not be hybridized to print summary.'.format(block.name)
            def _summary_hook(block, _, outputs):
                class_name = block.__class__.__name__
                block_idx = len(summary) - 1

                m_key = '%s-%i' % (class_name, block_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['output_shape'] = _get_shape_str(outputs)

                params = 0
                summary[m_key]['trainable'] = 0
                summary[m_key]['shared'] = 0
                for p in block.params.values():
                    params += p.data().size
                    summary[m_key]['trainable'] += 0 if p.grad_req == 'null' else p.data().size
                    if p in seen:
                        summary[m_key]['shared'] += p.data().size
                    else:
                        seen.add(p)
                summary[m_key]['n_params'] = params

            from .nn.basic_layers import Sequential, HybridSequential
            if not isinstance(block, (Sequential, HybridSequential)):
                hooks.append(block.register_forward_hook(_summary_hook))

        summary['Input'] = OrderedDict()
        summary['Input']['output_shape'] = _get_shape_str(inputs)
        summary['Input']['n_params'] = 0
        summary['Input']['trainable'] = 0
        summary['Input']['shared'] = 0

        try:
            self.apply(_register_summary_hook)
            self(*inputs)

            line_format = '{:>20}  {:>42} {:>15}'
            print('-'*80)
            print(line_format.format('Layer (type)', 'Output Shape', 'Param #'))
            print('='*80)
            total_params = 0
            trainable_params = 0
            shared_params = 0
            for layer in summary:
                print(line_format.format(layer,
                                         str(summary[layer]['output_shape']),
                                         summary[layer]['n_params']))
                total_params += summary[layer]['n_params']
                trainable_params += summary[layer]['trainable']
                shared_params += summary[layer]['shared']
            print('='*80)
            print('Parameters in forward computation graph, duplicate included')
            print('   Total params: ' + str(total_params))
            print('   Trainable params: ' + str(trainable_params))
            print('   Non-trainable params: ' + str(total_params - trainable_params))
            print('Shared params in forward computation graph: ' + str(shared_params))
            print('Unique parameters in model: ' + str(total_params - shared_params))
            print('-'*80)
        finally:
            for h in hooks:
                h.detach()


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
        <https://mxnet.io/tutorials/gluon/hybrid.html>`_
    """
    def __init__(self, prefix=None, params=None):
        super(HybridBlock, self).__init__(prefix=prefix, params=params)
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = []
        self._out_format = None
        self._in_format = None
        self._active = False
        self._flags = []
        self._callback = None
        self._monitor_all = False
        self._backend = None
        self._backend_opts = {}

    def __setattr__(self, name, value):
        """Registers parameters."""
        super(HybridBlock, self).__setattr__(name, value)
        if isinstance(value, HybridBlock):
            self._clear_cached_op()

    def _get_graph(self, *args):
        if not self._cached_graph:
            flatten_args, self._in_format = _flatten(args, "input")
            flatten_inputs = []
            symbol_inputs = []
            cnt = 0
            real_arg_num = sum([ele is not None for ele in flatten_args])
            if real_arg_num == 0:
                raise ValueError('All args are None and we do not support such a case.'
                                 ' Received args={}'.format(args))
            for arg in flatten_args:
                if arg is not None:
                    if real_arg_num > 1:
                        arg_sym = symbol.var('data{}'.format(cnt))
                    else:
                        arg_sym = symbol.var('data')
                    if isinstance(arg, _mx_np.ndarray):
                        arg_sym = arg_sym.as_np_ndarray()
                    cnt += 1
                    flatten_inputs.append(arg_sym)
                    symbol_inputs.append(arg_sym)
                else:
                    flatten_inputs.append(None)
            grouped_inputs = _regroup(flatten_inputs, self._in_format)
            params = {i: j.var() for i, j in self._reg_params.items()}
            with self.name_scope():
                out = self.hybrid_forward(symbol, *grouped_inputs, **params)  # pylint: disable=no-value-for-parameter
            out, self._out_format = _flatten(out, "output")

            self._cached_graph = symbol_inputs, symbol.Group(out, _check_same_symbol_type(out))

        return self._cached_graph

    def _build_cache(self, *args):
        data, out = self._get_graph(*args)
        data_names = {data.name: i for i, data in enumerate(data)}
        input_names = out.list_inputs()
        expected_names = set(input_names)

        # try to reuse cached_op_args for params
        if len(self._cached_op_args) > 0:
            params = {param_tuple[1].name:param_tuple[1]
                      for param_tuple in self._cached_op_args
                      if isinstance(param_tuple[1], Parameter)}
        else:
            params = self.collect_params()
        param_names = set(params.keys())
        for name in expected_names:
            assert name in param_names or name in data_names, \
                "Unknown input to HybridBlock: %s" %name

        used_data_names = [i for i in data_names if i in expected_names]
        if len(used_data_names) != len(data_names):
            unused = ', '.join(['%d-th'%i for name, i in data_names.items()
                                if name not in expected_names])
            warnings.warn("The %s input to HybridBlock is not used by any "
                          "computation. Is this intended?"%unused, stacklevel=4)

        used_param_names = [i for i in param_names if i in expected_names]
        if len(used_param_names) != len(param_names):
            unused = ', '.join(list(param_names - set(used_param_names)))
            warnings.warn("Parameter %s is not used by any computation. "
                          "Is this intended?"%unused, stacklevel=4)

        args, _ = _flatten(args, "input")
        try:
            for name in input_names:
                if name in params:
                    params[name].data()
        except DeferredInitializationError:
            self._deferred_infer_shape(*args)
            for name in input_names:
                if name in params:
                    params[name]._finish_deferred_init()

        arg_dict, aux_dict = dict(), dict()
        if self._backend:
            # set context for inputs
            _, _, ctx_set, _ = _gather_type_ctx_info(list(args))
            ctx = ctx_set.pop() if len(ctx_set) > 0 else None
            # get list of params in the order of out.list_arguments
            input_shapes = dict()
            for name in out.list_arguments():
                if name in data_names.keys() and data_names[name] < len(args):
                    if isinstance(args[data_names[name]], NDArray):
                        arg_dict[name] = args[data_names[name]]
                    elif (isinstance(args[data_names[name]], symbol.Symbol) and
                          '__shape__' in args[data_names[name]].list_attr()):
                        shape_str = args[data_names[name]].list_attr()['__shape__']
                        input_shapes[name] = tuple(map(int, shape_str.strip('()').split(',')))
                elif name in params:
                    arg_dict[name] = params[name].data()

            for name in out.list_auxiliary_states():
                if name in data_names.keys() and data_names[name] < len(args):
                    if isinstance(args[data_names[name]], NDArray):
                        aux_dict[name] = args[data_names[name]]
                    elif (isinstance(args[data_names[name]], symbol.Symbol) and
                          '__shape__' in args[data_names[name]].list_attr()):
                        shape_str = args[data_names[name]].list_attr()['__shape__']
                        input_shapes[name] = tuple(map(int, shape_str.strip('()').split(',')))
                elif name in params:
                    aux_dict[name] = params[name].data()

            # Partition the graph
            out = out.optimize_for(self._backend, arg_dict, aux_dict, ctx, input_shapes, **self._backend_opts)

            # convert to numpy symbol if needed
            if _mx_npx.is_np_array():
                out = out.as_np_ndarray()

            #update cached graph with partitioned graph
            self._cached_graph = data, out

        input_names = out.list_inputs()
        data_indices = []
        param_indices = []

        # In the default case, _cached_ops_args contains all the parameters from params (the sets are identical)
        # In the case of Partition API optimized graph _cached_ops_args might contain some parameters from params,
        # might contain some new parameters created during optimization and added to `arg_dict/aux_dict`,
        # and might not contain some parameters that were deleted during optimization.
        self._cached_op_args = []
        for i, name in enumerate(input_names):
            pair = None
            if name in data_names:
                data_indices.append(i)
                pair = (True, data_names[name])
            else:
                param_indices.append(i)
                if name in params:
                    param = params[name]
                else:
                    # The param is missing from the original params dictionary, which means the param must have
                    # been added by the Partition API backend
                    if name in arg_dict or name:
                        param_data = arg_dict[name]
                    elif name in aux_dict:
                        param_data = aux_dict[name]
                    else:
                        raise RuntimeError('A parameter was added to the graph during optimization but it was not '
                                           'added to the parameter dicts.\n'
                                           'Please check the backend.')

                    param = Parameter(name, dtype=param_data.dtype)
                    param._load_init(param_data, param_data.context)
                pair = (False, param)

            self._cached_op_args.append(pair)

        flags = [('data_indices', data_indices), ('param_indices', param_indices)] + \
                self._flags

        self._cached_op = ndarray.CachedOp(out, flags)


    def _deferred_infer_shape(self, *args):
        try:
            self.infer_shape(*args)
        except Exception as e:
            error_msg = "Deferred initialization failed because shape"\
                        " cannot be inferred. {}".format(e)
            raise ValueError(error_msg)

    def _call_cached_op(self, *args):
        if self._cached_op is None:
            self._build_cache(*args)
        assert self._cached_op, "Gluon failed to build the cache. " \
                                "This should never happen. " \
                                "Please submit an issue on Github" \
                                " https://github.com/apache/incubator-mxnet."
        if self._callback:
            self._cached_op._register_op_hook(self._callback, self._monitor_all)
            if len(self._flags) >= 2 and (self._flags[1] or self._flags[0]):
                warnings.warn("register_op_hook is experimental when static_alloc=True / static_shape=True "
                              " and may not work correctly")

        args, fmt = _flatten(args, "input")
        if fmt != self._in_format:
            # Do not raise in the case that the fmt or stored_fmt ends with None and
            # We are relying on the default values.
            if len(self._in_format) > len(fmt):
                valid = all([self._in_format[i] == -1
                             for i in range(len(fmt), len(self._in_format))])
                valid = valid and (fmt == self._in_format[:len(fmt)])
            elif len(self._in_format) < len(fmt):
                valid = all([fmt[i] == -1
                             for i in range(len(self._in_format), len(fmt))])
                valid = valid and (fmt[:len(self._in_format)] == self._in_format)
            else:
                valid = False
            if not valid:
                raise ValueError("The argument structure of HybridBlock does not match"
                                 " the cached version. Stored format = {}, input format = {}"
                                 .format(fmt, self._in_format))

        args_without_none = [ele for ele in args if ele is not None]
        cargs = [args_without_none[i] if is_arg else i.data()
                 for is_arg, i in self._cached_op_args]
        out = self._cached_op(*cargs)
        if isinstance(out, NDArray):
            out = [out]
        return _regroup(out, self._out_format)

    def optimize_for(self, x, *args, backend=None, clear=False,
                     static_alloc=False,
                     static_shape=False,
                     inline_limit=2,
                     forward_bulk_size=None,
                     backward_bulk_size=None,
                     **kwargs):
        """Partitions the current HybridBlock and optimizes it for a given backend
        without executing a forward pass. Modifies the HybridBlock in-place.

        Immediately partitions a HybridBlock using the specified backend. Combines
        the work done in the hybridize API with part of the work done in the forward
        pass without calling the CachedOp. Can be used in place of hybridize,
        afterwards `export` can be called or inference can be run. See README.md in
        example/extensions/lib_subgraph/README.md for more details.

        Examples
        --------
        # partition and then export to file
        block.optimize_for(x, backend='myPart')
        block.export('partitioned')

        # partition and then run inference
        block.optimize_for(x, backend='myPart')
        block(x)

        Parameters
        ----------
        x : NDArray
            first input to model
        *args : NDArray
            other inputs to model
        backend : str
            The name of backend, as registered in `SubgraphBackendRegistry`, default None
        clear : bool, default False
            Clears any previous optimizations
        static_alloc : bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        static_shape : bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
        inline_limit : optional int, default 2
            Maximum number of operators that can be inlined.
        forward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        backward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        **kwargs: The backend options, optional
            Passed on to `PrePartition` and `PostPartition` functions of `SubgraphProperty`
        """
        if len(kwargs) > 0:
            self._backend_opts = kwargs
        if not backend:
            raise ValueError('Must specify "backend" to optimize_for')

        self.hybridize(True, backend, clear, static_alloc, static_shape,
                       inline_limit, forward_bulk_size, backward_bulk_size)

        # do part of forward API call
        has_symbol, has_ndarray, ctx_set, _ = _gather_type_ctx_info([x] + list(args))
        if not has_symbol and not has_ndarray:
            raise ValueError('In HybridBlock, there must be one NDArray or one Symbol in the input.'
                             ' Please check the type of the args.\n')
        if len(ctx_set) > 1:
            raise ValueError('Found multiple contexts in the input, '
                             'After hybridized, the HybridBlock only supports one input '
                             'context. You can print the ele.ctx in the '
                             'input arguments to inspect their contexts. '
                             'Find all contexts = {}'.format(ctx_set))

        self._build_cache(x, *args)
        assert self._cached_op, "Gluon failed to build the cache. " \
                                "This should never happen. " \
                                "Please submit an issue on Github" \
                                " https://github.com/apache/incubator-mxnet."
        # do not actually call the cached_op

    def _clear_cached_op(self):
        self._cached_graph = ()
        self._cached_op = None
        self._cached_op_args = []

    def register_child(self, block, name=None):
        if not isinstance(block, HybridBlock):
            raise ValueError(
                "Children of HybridBlock must also be HybridBlock, " \
                "but %s has type %s. If you are using Sequential, " \
                "please try HybridSequential instead."%(
                    str(block), str(type(block))))
        super(HybridBlock, self).register_child(block, name)
        self._clear_cached_op()

    def hybridize(self, active=True, backend=None, clear=True,
                  static_alloc=False, static_shape=False,
                  inline_limit=2,
                  forward_bulk_size=None,
                  backward_bulk_size=None,
                  **kwargs):
        """Activates or deactivates :py:class:`HybridBlock` s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        backend : str
            The name of backend, as registered in `SubgraphBackendRegistry`, default None
        clear : bool, default True
            Clears any previous optimizations
        static_alloc : optional bool, default False
            Statically allocate memory to improve speed. Memory usage may increase.
        static_shape : optional bool, default False
            Optimize for invariant input shapes between iterations. Must also
            set static_alloc to True. Change of input shapes is still allowed
            but slower.
        inline_limit : optional int, default 2
            Maximum number of operators that can be inlined.
        forward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        backward_bulk_size : optional int, default None
            Segment size of bulk execution during forward pass.
        **kwargs:  optional
            Backend options.
        """
        if len(kwargs) > 0:
            self._backend_opts = kwargs

        self._backend = backend

        self._active = active
        self._flags = [("static_alloc", static_alloc), ("static_shape", static_shape),
                       ("inline_limit", inline_limit)]
        if forward_bulk_size is not None:
            self._flags.append(("forward_bulk_size", forward_bulk_size))
        if backward_bulk_size is not None:
            self._flags.append(("backward_bulk_size", backward_bulk_size))
        if clear:
            self._clear_cached_op()
        if active and self._forward_hooks or self._forward_pre_hooks:
            warnings.warn('"{block}" is being hybridized while still having forward hook/pre-hook. '
                          'If "{block}" is a child of HybridBlock, the hooks will not take effect.'
                          .format(block=self))
        super(HybridBlock, self).hybridize(active,
                                           static_alloc=static_alloc,
                                           static_shape=static_shape,
                                           inline_limit=inline_limit,
                                           forward_bulk_size=forward_bulk_size,
                                           backward_bulk_size=backward_bulk_size)

    def cast(self, dtype):
        self._clear_cached_op()
        super(HybridBlock, self).cast(dtype)

    def _infer_attrs(self, infer_fn, attr, *args):
        """Generic infer attributes."""
        inputs, out = self._get_graph(*args)
        args, _ = _flatten(args, "input")
        args_without_none = [ele for ele in args if ele is not None]
        with warnings.catch_warnings(record=True) as w:
            arg_attrs, _, aux_attrs = getattr(out, infer_fn)(
                **{i.name: getattr(j, attr) for i, j in zip(inputs, args_without_none)})
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

    def export(self, path, epoch=0, remove_amp_cast=True):
        """Export HybridBlock to json format that can be loaded by
        `gluon.SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface.

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
        sym.save('%s-symbol.json'%path, remove_amp_cast=remove_amp_cast)

        arg_names = set(sym.list_arguments())
        aux_names = set(sym.list_auxiliary_states())
        arg_dict = {}
        for is_arg, param in self._cached_op_args:
            if not is_arg:
                name = param.name
                if name in arg_names:
                    arg_dict['arg:{}'.format(name)] = param._reduce()
                else:
                    if name not in aux_names:
                        warnings.warn('Parameter "{name}" is not found in the graph. '
                                      .format(name=name), stacklevel=3)
                    else:
                        arg_dict['aux:%s'%name] = param._reduce()
        save_fn = _mx_npx.save if is_np_array() else ndarray.save
        save_fn('%s-%04d.params'%(path, epoch), arg_dict)

    def register_op_hook(self, callback, monitor_all=False):
        """Install op hook for block recursively.

        Parameters
        ----------
        callback : function
            Takes a string and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input and output, otherwise monitor output only.
        """
        self._callback = callback
        self._monitor_all = monitor_all
        for cld in self._children.values():
            cld._callback = callback
            cld._monitor_all = monitor_all

    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""

        has_symbol, has_ndarray, ctx_set, first_ctx = _gather_type_ctx_info([x] + list(args))
        if has_symbol and has_ndarray:
            raise ValueError('In HybridBlock, we do not support mixed NDArrays and Symbols'
                             ' types for the input. Please check the type of the args.\n')
        if not has_symbol and not has_ndarray:
            raise ValueError('In HybridBlock, there must be one NDArray or one Symbol in the input.'
                             ' Please check the type of the args.\n')
        if has_ndarray:
            ctx = first_ctx
            if self._active:
                if len(ctx_set) > 1:
                    raise ValueError('Find multiple contexts in the input, '
                                     'After hybridized, the HybridBlock only supports one input '
                                     'context. You can print the ele.ctx in the '
                                     'input arguments to inspect their contexts. '
                                     'Find all contexts = {}'.format(ctx_set))
                with ctx:
                    return self._call_cached_op(x, *args)
            with ctx:
                try:
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                except DeferredInitializationError:
                    self._deferred_infer_shape(x, *args)
                    for _, v in self.params.items():
                        v._finish_deferred_init()
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}

                return self.hybrid_forward(ndarray, x, *args, **params)
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

def _common_prefix(names):
    """Get the common prefix for all names"""
    if not names:
        return ''
    prefix = names[0]
    for name in names:
        i = 0
        while i < len(prefix) and i < len(name) and prefix[i] == name[i]:
            i += 1
        prefix = prefix[:i]
    return prefix


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
    def imports(symbol_file, input_names, param_file=None, ctx=None, allow_missing=False,
                ignore_extra=False):
        """Import model previously saved by `gluon.HybridBlock.export` or
        `Module.save_checkpoint` as a `gluon.SymbolBlock` for use in Gluon.

        Parameters
        ----------
        symbol_file : str
            Path to symbol file.
        input_names : list of str
            List of input variable names
        param_file : str, optional
            Path to parameter file.
        ctx : Context, default None
            The context to initialize `gluon.SymbolBlock` on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.

        Returns
        -------
        gluon.SymbolBlock
            `gluon.SymbolBlock` loaded from symbol and parameter files.

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
        if is_np_array():
            sym = np_symbol.load(symbol_file)
        else:
            sym = symbol.load(symbol_file)
        if isinstance(input_names, str):
            input_names = [input_names]
        if param_file is None:
            # Get a valid type inference by using fp32
            inputs = [symbol.var(i, dtype=mx_real_t) for i in input_names]
        else:
            # Do not specify type, rely on saved params type instead
            inputs = [symbol.var(i).as_np_ndarray() if is_np_array() else symbol.var(i) for i in input_names]
        ret = SymbolBlock(sym, inputs)
        if param_file is not None:
            ret.collect_params().load(param_file, ctx, allow_missing, ignore_extra, cast_dtype=True,
                                      dtype_source='saved')
        return ret

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['{block} : {numinputs} -> {numoutputs}'.format(block=self._cached_graph[1],
                                                                           numinputs=len(self._cached_graph[0]),
                                                                           numoutputs=len(self._cached_graph[1].
                                                                                          list_outputs()))])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

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
        input_names = set()
        for i in syms:
            assert len(i.get_internals().list_outputs()) == 1, \
                "Input symbols must be variable, but %s is an output of operators"%str(i)
            input_names.add(i.name)

        # check if any symbol is row_sparse
        row_sparse_storage = ndarray.ndarray._STORAGE_TYPE_STR_TO_ID['row_sparse']

        for i in out:
            for j in i.get_internals():
                assert(j.attr("__storage_type__") != str(row_sparse_storage)), \
                    "SymbolBlock doesn't support Parameter '%s' because its storage " \
                    "type is 'row_sparse'." % j.name
        if len(out) > 1:
            out = symbol.Group(out, _check_same_symbol_type(out))
        else:
            out = out[0]

        # Infer type of parameters. Without this, every parameter will be created with
        # default type i.e., fp32
        arg_params = out.list_arguments()
        aux_params = out.list_auxiliary_states()

        arg_types, aux_types = _infer_param_types(syms, out, arg_params, aux_params)

        for i, arg in enumerate(arg_params):
            if arg not in input_names:
                self.params.get(arg, allow_deferred_init=True, dtype=arg_types[i])

        for i, aux in enumerate(aux_params):
            if aux not in input_names:
                self.params.get(aux, grad_req='null', allow_deferred_init=True, dtype=aux_types[i])

        self._cached_graph = syms, out
        len_prefix = len(_common_prefix(list(self._params.keys())))
        self._reg_params = {key[len_prefix:]: val for key, val in self._params.items()}

    def forward(self, x, *args):
        if isinstance(x, NDArray):
            with x.ctx:
                return self._call_cached_op(x, *args)

        assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
        args, in_fmt = _flatten([x] + list(args), "input")
        assert in_fmt == self._in_format, "Invalid input format"
        ret = copy.copy(self._cached_graph[1])
        ret._compose(**{k.name: v for k, v in zip(self._cached_graph[0], args)})
        return _regroup(list(ret), self._out_format)

    def _clear_cached_op(self):
        tmp = self._cached_graph
        super(SymbolBlock, self)._clear_cached_op()
        self._cached_graph = tmp

    def cast(self, dtype):
        self._clear_cached_op()
        super(SymbolBlock, self).cast(dtype)
        if np.dtype(dtype).name == 'float16':
            # correct BatchNorm types back to float32 due to its special requirement
            out = self._cached_graph[1]
            params_list = out.get_internals().list_inputs()
            for node in params_list:
                if node.endswith('running_var'):
                    prefix = node[:-11]
                    sibs = [prefix + t for t in ('running_mean', 'gamma', 'beta')]
                    is_bn = all(p in params_list for p in sibs)
                    if is_bn:
                        self.params.get(node).cast('float32')
                        for sib in sibs:
                            self.params.get(sib).cast('float32')
                if node.endswith('moving_var'):
                    # another convention used
                    prefix = node[:-10]
                    sibs = [prefix + t for t in ('moving_mean', 'gamma', 'beta')]
                    is_bn = all(p in params_list for p in sibs)
                    if is_bn:
                        self.params.get(node).cast('float32')
                        for sib in sibs:
                            self.params.get(sib).cast('float32')

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError

    def reset_ctx(self, ctx):
        """Re-assign all Parameters to other contexts. If the Block is hybridized, it will reset the _cached_op_args.
        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
        params = self.collect_params()
        if self._cached_op:
            for p in self._cached_op_args:
                # resetting parameters creating by the partitioning backend
                if p.name not in params:
                    p.reset_ctx(ctx)
        for p in params.values():
            p.reset_ctx(ctx)

def _infer_param_types(in_params, out_params, arg_params, aux_params, default_dtype=mx_real_t):
    """Utility function that helps in inferring DType of args and auxs params
    from given input param.

    Parameters
    ----------
    in_params: List of Symbol
        List of input symbol variables.
    out_params: Symbol
        Output symbol variable.
    arg_params: List of Str
        List of names of argument parametrs.
    aux_params: List of Str
        List of names of auxiliary parameters.
    default_dtype: numpy.dtype or str, default 'float32'
        Default data type for arg_params and aux_params, if unable to infer the type.

    Returns
    -------
    arg_types: List of numpy.dtype
        List of arg_params type. Order is same as arg_params.
        Defaults to 'float32', if unable to infer type.
    aux_types: List of numpy.dtype
        List of aux_params type. Order is same as aux_params.
        Defaults to 'float32', if unable to infer type.
    """
    arg_types = None
    aux_types = None

    # Get Input symbol details. This will be used to infer types of
    # other parameters.
    input_sym_names = [in_param.name for in_param in in_params]

    # Try to infer input types. If not successful, we will set default dtype.
    # If successful, we will try to infer other params in the graph.
    input_sym_arg_types = []
    can_infer_input_type = True
    for in_param in in_params:
        input_sym_arg_type = in_param.infer_type()[0]
        if not input_sym_arg_type or len(input_sym_arg_type) < 1:
            can_infer_input_type = False
            break
        else:
            input_sym_arg_types.append(in_param.infer_type()[0][0])

    # Try to infer types of other parameters.
    if can_infer_input_type:
        params = {k:v for k, v in zip(input_sym_names, input_sym_arg_types)}
        try:
            arg_types, _, aux_types = out_params.infer_type(**params)
        except MXNetError:
            # Cannot infer type with current input
            arg_types, aux_types = None, None

    if arg_types is None or len(arg_types) != len(arg_params):
        arg_types = []
        for _ in arg_params:
            arg_types.append(default_dtype)

    if aux_types is None or len(aux_types) != len(aux_params):
        aux_types = []
        for _ in aux_params:
            aux_types.append(default_dtype)

    return (arg_types, aux_types)
