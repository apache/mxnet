# coding: utf-8
# pylint: disable= arguments-differ
"""Neural network layers."""

from ..layer import Layer, HybridLayer


class Sequential(Layer):
    """Stack Layers sequentially.

    Example
    -------
    >>> net = nn.Sequential()
    >>> with net.name_scope():
    ...     net.add(Dense(10, activation='relu'))
    ...     net.add(Dense(20))
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

    Example
    -------
    >>> net = nn.HSequential()
    >>> with net.name_scope():
    ...     net.add(Dense(10, activation='relu'))
    ...     net.add(Dense(20))
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
    weight_initializer: Initializer for the `kernel` weights matrix
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


    Input shape:
        a 2D input with shape `(batch_size, in_units)`.

    Output shape:
        the output would have shape `(batch_size, units)`.
    """
    def __init__(self, units, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer=None,
                 in_units=0, **kwargs):
        super(Dense, self).__init__(**kwargs)
        with self.name_scope():
            self._units = units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer)
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
    """Applies an activation function to input. Refer
    `mxnet.ndarray.Activation <http://mxnet.io/api/python/ndarray.html#mxnet.ndarray.Activation>`_
    to learn more.

    Parameters
    ----------
    activation: name of activation function to use
        See: help on Activation operator


    Input shape:
        Arbitrary.

    Output shape:
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
    which helps prevent overfitting. Refer
    `mxnet.ndarray.Dropout <http://mxnet.io/api/python/ndarray.html#mxnet.ndarray.Dropout>`_
    to learn more.

    Parameters
    ----------
    rate: float between 0 and 1. Fraction of the input units to drop.

    References
    ----------
        `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_
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
    close to 0 and the activation standard deviation close to 1. Refer
    `mxnet.ndarray.BatchNorm <http://mxnet.io/api/python/ndarray.html#mxnet.ndarray.BatchNorm>`_
    to learn more.

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
                 beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 num_features=0, **kwargs):
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

    It allows a small gradient when the unit is not active::

        `f(x) = alpha * x for x < 0`,
        `f(x) = x for x >= 0`.

    Refer
    `mxnet.ndarray.LeakyReLU <http://mxnet.io/api/python/ndarray.html#mxnet.ndarray.LeakyReLU>`_
    to learn more.

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

    Refer
    `mxnet.ndarray.Embedding <http://mxnet.io/api/python/ndarray.html#mxnet.ndarray.Embedding>`_
    to learn more.

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : Initializer
        Initializer for the `embeddings` matrix


    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    """
    def __init__(self, input_dim, output_dim, dtype='float32',
                 weight_initializer=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=weight_initializer)

    def hybrid_forward(self, F, x, weight):
        return F.Embedding(x, weight, **self._kwargs)
