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
                    self.dense0 = nn.Dense(20, in_units=10)
                    self.dense1 = nn.Dense(20, in_units=20)

            def forward(self, F, x):
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
        self._reg_params = {}


    def __setattr__(self, name, value):
        """Registers parameters."""
        super(Layer, self).__setattr__(name, value)
        if isinstance(value, Parameter):
            self._reg_params[name] = value
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

    def infer_shape(self, *args):
        """Infer parameter shape given input shapes.

        *args : list of tuple
            A list of input argument shapes.
        """
        inputs = [symbol.var('__input%d__'%i, shape=shape)
                  for i, shape in enumerate(args)]
        params = {k: v.var() for k, v in self._reg_params.items()}
        sym = self.forward(symbol, *inputs, **params)
        arg_shapes, _, aux_shapes = sym.infer_shape()
        sdict = {name: shape for name, shape in zip(sym.list_arguments(), arg_shapes)}
        sdict.update(
            {name : shape for name, shape in zip(sym.list_auxiliary_states(), aux_shapes)})
        for i in self.params.values():
            i.shape = sdict[i.name]

    def __call__(self, *args):
        """Call forward."""
        return self.call(*args)  # pylint: disable=no-value-for-parameter

    def call(self, x, *args):
        """Defines the forward computation. Arguments can be either NDArray or Symbol."""
        if isinstance(x, NDArray):
            with x.context as ctx:
                try:
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                except DeferredInitializationError:
                    arg_shapes = [x.shape]
                    arg_shapes += [i.shape if isinstance(i, NDArray) else i for i in args]
                    self.infer_shape(*arg_shapes)
                    for i in self.params.values():
                        i._finish_deferred_init()
                    params = {k: v.data(ctx) for k, v in self._reg_params.items()}
                return self.forward(ndarray, x, *args, **params)
        else:
            assert isinstance(x, Symbol), \
                "Layer requires the first argument to forward be either " \
                "Symbol or NDArray, but got %s"%type(x)
            params = {k: v.var() for k, v in self._reg_params.items()}
            return self.forward(symbol, x, *args, **params)

    def forward(self, F, x, *args, **kwargs):
        """Simple forward supports both `Symbol` and `NDArray` API.

        Parameters
        ----------
        F : {mxnet.ndarray, mxnet.symbol}
            Name space of operators. `F` will be set to `mx.sym` when x is `Symbol`
            instance and `mx.nd` when x is `NDArray` instance.
        x : NDArray or Symbol
            The first input tensor.
        *args : list of NDArray or list of Symbol
            Additional input tensors.
        **kwargs : dict of str to NDArray or dict of str to Symbol
            `Symbol` or `NDArray` value of registered Parameters.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class Sequential(Layer):
    """Stack Layers sequentially.

    Example::
        net = nn.Sequential()
        net.add(Dense(10, activation='relu'))
        net.add(Dense(20))
    """
    def __init__(self):
        super(Sequential, self).__init__(prefix='', params=None)

    def add(self, layer):
        """Add layer on top of the stack."""
        self.register_child(layer)

    def call(self, x):
        #pylint: disable=arguments-differ
        for layer in self._children:
            x = layer(x)
        return x

    def forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use flatten to convert it
    to rank 2 manually if necessary.

    Example::
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, in_uints=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # No need to specify the size of the input if you only want to
        # use the `Symbol` API:
        model = Sequential()
        model.add(Dense(32))

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
            self._op = symbol.CachedOp('FullyConnected', 3 if use_bias else 2,
                                       num_hidden=units, no_bias=not use_bias)
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=kernel_initializer)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer)
            if activation is not None:
                self.act = Activation(activation)
            else:
                self.act = None

    def forward(self, F, x, weight, bias=None):
        if bias is None:
            act = F.invoke(self._op, [x, weight])
        else:
            act = F.invoke(self._op, [x, weight, bias])
        if self.act is not None:
            act = self.act(act)
        return act


class Activation(Layer):
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
        self._op = symbol.CachedOp('Activation', 1, act_type=self._act_type)

    def _alias(self):
        return self._act_type

    def forward(self, F, x):
        return F.invoke(self._op, [x])


class Dropout(Layer):
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
        self._op = symbol.CachedOp('Dropout', 1, p=rate)

    def forward(self, F, x):
        return F.invoke(self._op, [x])


class BatchNorm(Layer):
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
        assert axis == 1, \
            "Only support NC* layout, i.e. channel must be in the second dimension"
        attrs = {'eps': epsilon, 'momentum': momentum, 'fix_gamma': not center}
        self._op = symbol.CachedOp('BatchNorm', 5, **attrs)

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

    def forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.invoke(self._op, [x, gamma, beta, running_mean, running_var])


class LeakyReLU(Layer):
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
        self._op = symbol.CachedOp('LeakyReLU', 1, act_type='leaky', slope=alpha)

    def forward(self, F, x):
        return F.invoke(self._op, [x])


class Embedding(Layer):
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
        self._op = symbol.CachedOp('Embedding', 2, input_dim=input_dim,
                                   output_dim=output_dim, dtype=dtype)
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=embeddings_initializer)

    def forward(self, F, x, weight):
        return F.invoke(self._op, [x, weight])
