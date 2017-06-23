# coding: utf-8
# pylint: disable= arguments-differ
"""Neural network layers."""

from ... import symbol, ndarray
from ...symbol import Symbol
from ...ndarray import NDArray
from ... import name as _name
from ..parameter import Parameter, ParameterDict, DeferredInitializationError


class _LayerScope(object):
    """Scope for collecting sub-layers."""
    _current = None

    def __init__(self, layer):
        self._layer = layer
        self._counter = {}
        self._old_scope = None

    @staticmethod
    def create_prefix(prefix, hint):
        if _LayerScope._current is None:
            if prefix is None:
                return _name.NameManager.current.get(None, hint) + '_'
            return prefix
        else:
            if prefix is None:
                count = _LayerScope._current._counter.get(hint, 0)
                prefix = '%s%d_'%(hint, count)
                _LayerScope._current._counter[hint] = count + 1
            return _LayerScope._current._layer.prefix+prefix

    @staticmethod
    def create_params(prefix, params):
        if params is not None:
            return ParameterDict(params.prefix, params)
        if _LayerScope._current is not None:
            return ParameterDict(prefix, _LayerScope._current._layer._params._shared)
        return ParameterDict(prefix)

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
        self._prefix = _LayerScope.create_prefix(prefix, self._alias())
        self._params = _LayerScope.create_params(self._prefix, params)
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
            if isinstance(layer, Sequantial):
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
        arg_shapes, _, aux_shapes = out.infer_shape(
            **{i.name: j.shape for i, j in zip(syms, args)})
        sdict = {i: j for i, j in zip(out.list_arguments(), arg_shapes)}
        sdict.update(
            {name : shape for name, shape in zip(out.list_auxiliary_states(), aux_shapes)})
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


class Sequential(Layer):
    """Stack Layers sequentially.

    Example::
        net = nn.Sequential()
        with net.name_scope():
            net.add(Dense(10, activation='relu'))
            net.add(Dense(20))
    """
    def __init__(self, prefix=None, params=None):
        super(Sequential, self).__init__(prefix=prefix, params=params)

    def add(self, layer):
        """Add layer on top of the stack."""
        self.register_child(layer)

    def forward(self, x):
        for layer in self._children:
            x = layer(x)
        return x


class HSequential(HybridLayer):
    """Stack HybridLayers sequentially.

    Example::
        net = nn.HSequential()
        with net.name_scope():
            net.add(Dense(10, activation='relu'))
            net.add(Dense(20))
    """
    def __init__(self, prefix=None, params=None):
        super(HSequential, self).__init__(prefix=prefix, params=params)

    def add(self, layer):
        """Add layer on top of the stack."""
        self.register_child(layer)

    def hybrid_forward(self, F, x):
        for layer in self._children:
            x = layer(x)
        return x


class Dense(HybridLayer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use flatten to convert it
    to rank 2 manually if necessary.

    Parameters
    ----------
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
        (see help on Activation operator).
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
        (see mxnet.initializer).
    bias_initializer: Initializer for the bias vector
        (see mxnet.initializer).
    in_units : int
        Size of input data. No need to specify for `Symbol` API. But must be
        specified for every Dense layer if you want to use `NDArray` API.
    prefix : str or None
        See document of Layer.
    params : ParameterDict or None
        See document of Layer.

    Input shape
    -----------
    a 2D input with shape `(batch_size, in_units)`.

    Output shape
    ------------
    the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer=None, bias_initializer=None,
                 in_units=0, **kwargs):
        super(Dense, self).__init__(**kwargs)
        with self.name_scope():
            self._units = units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=kernel_initializer)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer)
            else:
                self.bias = None
            if activation is not None:
                self.act = Activation(activation)
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):
        if bias is None:
            act = F.FullyConnected(x, weight, no_bias=True, num_hidden=self._units)
        else:
            act = F.FullyConnected(x, weight, bias, num_hidden=self._units)
        if self.act is not None:
            act = self.act(act)
        return act


class Activation(HybridLayer):
    """Applies an activation function to input.

    Parameters
    ----------
    activation: name of activation function to use
        See: help on Activation operator

    Input shape
    -----------
    Arbitrary.

    Output shape
    ------------
    Same shape as input.
    """
    def __init__(self, activation, **kwargs):
        self._act_type = activation
        super(Activation, self).__init__(**kwargs)

    def _alias(self):
        return self._act_type

    def hybrid_forward(self, F, x):
        return F.Activation(x, act_type=self._act_type)


class Dropout(HybridLayer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Parameters
    ----------
    rate: float between 0 and 1. Fraction of the input units to drop.

    References
    ----------
    [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
        http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self._rate = rate

    def hybrid_forward(self, F, x):
        return F.Dropout(x, p=self._rate)


class BatchNorm(HybridLayer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    """
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-3, center=True, scale=True,
                 num_features=0, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not center}

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(num_features,), init=gamma_initializer)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(num_features,), init=beta_initializer)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(num_features,),
                                            init=running_mean_initializer)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(num_features,),
                                           init=running_variance_initializer)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var, **self._kwargs)


class LeakyReLU(HybridLayer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

    Parameters
    ----------
    alpha: float
        Negative slope coefficient. Must be >= 0.
    """
    def __init__(self, alpha, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='leaky', slope=self._alpha)


class Embedding(HybridLayer):
    """Turns non-negative integers (indexes/tokens) into dense
    vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    embeddings_initializer : Initializer
        Initializer for the `embeddings` matrix

    Input shape
    -----------
    2D tensor with shape: `(batch_size, sequence_length)`.

    Output shape
    ------------
    3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    """
    def __init__(self, input_dim, output_dim, dtype='float32',
                 embeddings_initializer=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=embeddings_initializer)

    def hybrid_forward(self, F, x, weight):
        return F.Embedding(x, weight, **self._kwargs)
