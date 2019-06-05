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
"""Basic neural network layers."""
__all__ = ['Sequential', 'HybridSequential', 'Dense', 'Dropout', 'Embedding',
           'BatchNorm', 'InstanceNorm', 'LayerNorm', 'GroupNorm',
           'Flatten', 'Lambda', 'HybridLambda']
import warnings
import numpy as np

from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent, _to_classic_arrays, _to_np_arrays
from ... import nd, sym


class Sequential(Block):
    """Stacks Blocks sequentially.

    Example::

        net = nn.Sequential()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
    """
    def __init__(self, prefix=None, params=None):
        super(Sequential, self).__init__(prefix=prefix, params=params)

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)(prefix=self._prefix)
            with net.name_scope():
                net.add(*layers)
            return net
        else:
            return layers

    def __len__(self):
        return len(self._children)

    def hybridize(self, active=True, **kwargs):
        """Activates or deactivates `HybridBlock` s recursively. Has no effect on
        non-hybrid children.

        Parameters
        ----------
        active : bool, default True
            Whether to turn hybrid on or off.
        **kwargs : string
            Additional flags for hybridized operator.
        """
        if self._children and all(isinstance(c, HybridBlock) for c in self._children.values()):
            warnings.warn(
                "All children of this Sequential layer '%s' are HybridBlocks. Consider "
                "using HybridSequential for the best performance."%self.prefix, stacklevel=2)
        super(Sequential, self).hybridize(active, **kwargs)


class HybridSequential(HybridBlock):
    """Stacks HybridBlocks sequentially.

    Example::

        net = nn.HybridSequential()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
        net.hybridize()
    """
    def __init__(self, prefix=None, params=None):
        super(HybridSequential, self).__init__(prefix=prefix, params=params)

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    def hybrid_forward(self, F, x):
        for block in self._children.values():
            x = block(x)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)(prefix=self._prefix)
            with net.name_scope():
                net.add(*layers)
            return net
        else:
            return layers

    def __len__(self):
        return len(self._children)


class Dense(HybridBlock):
    r"""Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, weight) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `weight` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use `flatten` to convert it
    to rank 2 manually if necessary.

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    activation : str
        Activation function to use. See help on `Activation` layer.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    flatten: bool, default True
        Whether the input tensor should be flattened.
        If true, all but the first axis of input data are collapsed together.
        If false, all but the last axis of input data are kept the same, and the transformation
        applies on the last axis.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.


    Inputs:
        - **data**: if `flatten` is True, `data` should be a tensor with shape
          `(batch_size, x1, x2, ..., xn)`, where x1 * x2 * ... * xn is equal to
          `in_units`. If `flatten` is False, `data` should have shape
          `(x1, x2, ..., xn, in_units)`.

    Outputs:
        - **out**: if `flatten` is True, `out` will be a tensor with shape
          `(batch_size, units)`. If `flatten` is False, `out` will have shape
          `(x1, x2, ..., xn, units)`.
    """
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):
        # TODO(junwu): This is a temp solution to reuse legacy ops for np.ndarray.
        # We should rewrite this with np/npx ops.
        x, weight, bias = _to_classic_arrays(x, weight, bias)
        act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return _to_np_arrays(act)

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


class Dropout(HybridBlock):
    """Applies Dropout to the input.

    Dropout consists in randomly setting a fraction `rate` of input units
    to 0 at each update during training time, which helps prevent overfitting.

    Parameters
    ----------
    rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    axes : tuple of int, default ()
        The axes on which dropout mask is shared. If empty, regular dropout is applied.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.

    References
    ----------
        `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_
    """
    def __init__(self, rate, axes=(), **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self._rate = rate
        self._axes = axes

    def hybrid_forward(self, F, x):
        x = _to_classic_arrays(x)
        if self._rate > 0:
            out = F.Dropout(x, p=self._rate, axes=self._axes, name='fwd', cudnn_off=False)
        else:
            out = F.identity(x)
        return _to_np_arrays(out)

    def __repr__(self):
        s = '{name}(p = {_rate}, axes={_axes})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class BatchNorm(HybridBlock):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    running_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the running mean.
    running_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the running variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        if in_channels != 0:
            self.in_channels = in_channels

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(BatchNorm, self).cast(dtype)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var,
                           name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class Embedding(HybridBlock):
    r"""Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]

    Note: if `sparse_grad` is set to True, the gradient w.r.t weight will be
    sparse. Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
    and Adam. By default lazy updates is turned on, which may perform differently
    from standard updates. For more details, please check the Optimization API at:
    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e. maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : Initializer
        Initializer for the `embeddings` matrix.
    sparse_grad: bool
        If True, gradient w.r.t. weight will be a 'row_sparse' NDArray.

    Inputs:
        - **data**: (N-1)-D tensor with shape: `(x1, x2, ..., xN-1)`.

    Output:
        - **out**: N-D tensor with shape: `(x1, x2, ..., xN-1, output_dim)`.
    """
    def __init__(self, input_dim, output_dim, dtype='float32',
                 weight_initializer=None, sparse_grad=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        grad_stype = 'row_sparse' if sparse_grad else 'default'
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype, 'sparse_grad': sparse_grad}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=weight_initializer, dtype=dtype,
                                      allow_deferred_init=True, grad_stype=grad_stype)

    def hybrid_forward(self, F, x, weight):
        return F.Embedding(x, weight, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__,
                        **self._kwargs)


class Flatten(HybridBlock):
    r"""Flattens the input to two dimensional.

    Inputs:
        - **data**: input tensor with arbitrary shape `(N, x1, x2, ..., xn)`

    Output:
        - **out**: 2D tensor with shape: `(N, x1 \cdot x2 \cdot ... \cdot xn)`
    """
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.Flatten(x)

    def __repr__(self):
        return self.__class__.__name__


class InstanceNorm(HybridBlock):
    r"""
    Applies instance normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array where (n>2) and normalizes
    the input using the following formula:

    .. math::

      \bar{C} = \{i \mid i \neq 0, i \neq axis\}

      out = \frac{x - mean[data, \bar{C}]}{ \sqrt{Var[data, \bar{C}]} + \epsilon}
       * gamma + beta

    Parameters
    ----------
    axis : int, default 1
        The axis that will be excluded in the normalization process. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `InstanceNorm`. If `layout='NHWC'`, then set `axis=3`. Data will be
        normalized along axes excluding the first axis and the axis given.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.

    References
    ----------
        `Instance Normalization: The Missing Ingredient for Fast Stylization
        <https://arxiv.org/abs/1607.08022>`_

    Examples
    --------
    >>> # Input of shape (2,1,2)
    >>> x = mx.nd.array([[[ 1.1,  2.2]],
    ...                 [[ 3.3,  4.4]]])
    >>> # Instance normalization is calculated with the above formula
    >>> layer = InstanceNorm()
    >>> layer.initialize(ctx=mx.cpu(0))
    >>> layer(x)
    [[[-0.99998355  0.99998331]]
     [[-0.99998319  0.99998361]]]
    <NDArray 2x1x2 @cpu(0)>
    """
    def __init__(self, axis=1, epsilon=1e-5, center=True, scale=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon, 'axis': axis, 'center': center, 'scale': scale}
        self._axis = axis
        self._epsilon = epsilon
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, x, gamma, beta):
        if self._axis == 1:
            return F.InstanceNorm(x, gamma, beta,
                                  name='fwd', eps=self._epsilon)
        x = x.swapaxes(1, self._axis)
        return F.InstanceNorm(x, gamma, beta, name='fwd',
                              eps=self._epsilon).swapaxes(1, self._axis)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class LayerNorm(HybridBlock):
    r"""
    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

      out = \frac{x - mean[data, axis]}{ \sqrt{Var[data, axis] + \epsilon}} * gamma + beta

    Parameters
    ----------
    axis : int, default -1
        The axis that should be normalized. This is typically the axis of the channels.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.


    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.

    References
    ----------
        `Layer Normalization
        <https://arxiv.org/pdf/1607.06450.pdf>`_

    Examples
    --------
    >>> # Input of shape (2, 5)
    >>> x = mx.nd.array([[1, 2, 3, 4, 5], [1, 1, 2, 2, 2]])
    >>> # Layer normalization is calculated with the above formula
    >>> layer = LayerNorm()
    >>> layer.initialize(ctx=mx.cpu(0))
    >>> layer(x)
    [[-1.41421    -0.707105    0.          0.707105    1.41421   ]
     [-1.2247195  -1.2247195   0.81647956  0.81647956  0.81647956]]
    <NDArray 2x5 @cpu(0)>
    """
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, prefix=None, params=None):
        super(LayerNorm, self).__init__(prefix=prefix, params=params)
        self._kwargs = {'eps': epsilon, 'axis': axis, 'center': center, 'scale': scale}
        self._axis = axis
        self._epsilon = epsilon
        self._center = center
        self._scale = scale
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, data, gamma, beta):
        norm_data = F.LayerNorm(data, gamma=gamma, beta=beta, axis=self._axis, eps=self._epsilon)
        return norm_data

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class GroupNorm(HybridBlock):
    r"""
    Applies group normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array where the leftmost 2 axis are
    `batch` and `channel` respectively:

    .. math::

      x = x.reshape((N, num_groups, C // num_groups, ...))
      axis = (2, ...)
      out = \frac{x - mean[x, axis]}{ \sqrt{Var[x, axis] + \epsilon}} * gamma + beta

    Parameters
    ----------
    num_groups: int, default 1
        Number of groups to separate the channel axis into.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.


    Inputs:
        - **data**: input tensor with shape (N, C, ...).

    Outputs:
        - **out**: output tensor with the same shape as `data`.

    References
    ----------
        `Group Normalization
        <https://arxiv.org/pdf/1803.08494.pdf>`_

    Examples
    --------
    >>> # Input of shape (2, 3, 4)
    >>> x = mx.nd.array([[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]])
    >>> # Group normalization is calculated with the above formula
    >>> layer = GroupNorm()
    >>> layer.initialize(ctx=mx.cpu(0))
    >>> layer(x)
    [[[-1.5932543 -1.3035717 -1.0138891 -0.7242065]
      [-0.4345239 -0.1448413  0.1448413  0.4345239]
      [ 0.7242065  1.0138891  1.3035717  1.5932543]]
     [[-1.5932543 -1.3035717 -1.0138891 -0.7242065]
      [-0.4345239 -0.1448413  0.1448413  0.4345239]
      [ 0.7242065  1.0138891  1.3035717  1.5932543]]]
    <NDArray 2x3x4 @cpu(0)>
    """
    def __init__(self, num_groups=1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 prefix=None, params=None):
        super(GroupNorm, self).__init__(prefix=prefix, params=params)
        self._kwargs = {'eps': epsilon, 'num_groups': num_groups, 'center': center, 'scale': scale}
        self._num_groups = num_groups
        self._epsilon = epsilon
        self._center = center
        self._scale = scale
        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(num_groups,), init=gamma_initializer,
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(num_groups,), init=beta_initializer,
                                    allow_deferred_init=True)

    def hybrid_forward(self, F, data, gamma, beta):
        norm_data = F.GroupNorm(data, gamma=gamma, beta=beta, num_groups=self._num_groups, eps=self._epsilon)
        return norm_data

    def __repr__(self):
        s = '{name}({content})'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class Lambda(Block):
    r"""Wraps an operator or an expression as a Block object.


    Parameters
    ----------
    function : str or function
        Function used in lambda must be one of the following:
        1) the name of an operator that is available in ndarray. For example::

            block = Lambda('tanh')

        2) a function that conforms to ``def function(*args)``. For example::

            block = Lambda(lambda x: nd.LeakyReLU(x, slope=0.1))

    Inputs:
        - ** *args **: one or more input data. Their shapes depend on the function.

    Output:
        - ** *outputs **: one or more output data. Their shapes depend on the function.
    """
    def __init__(self, function, prefix=None):
        super(Lambda, self).__init__(prefix=prefix)
        if isinstance(function, str):
            assert hasattr(nd, function), \
                   "Function name %s is not found in ndarray." % function
            self._func_impl = getattr(nd, function)
        elif callable(function):
            self._func_impl = function
        else:
            raise ValueError(
                "Unrecognized function in lambda: {} of type {}"
                .format(function, type(function)))

    def forward(self, *args):
        return self._func_impl(*args)

    def __repr__(self):
        return '{name}({function})'.format(name=self.__class__.__name__,
                                           function=self._func_impl.__name__)


class HybridLambda(HybridBlock):
    r"""Wraps an operator or an expression as a HybridBlock object.

    Parameters
    ----------
    function : str or function
        Function used in lambda must be one of the following:
        1) The name of an operator that is available in both symbol and ndarray. For example::

            block = HybridLambda('tanh')

        2) A function that conforms to ``def function(F, data, *args)``. For example::

            block = HybridLambda(lambda F, x: F.LeakyReLU(x, slope=0.1))

    Inputs:
        - ** *args **: one or more input data. First argument must be symbol or ndarray. Their \
            shapes depend on the function.

    Output:
        - ** *outputs **: one or more output data. Their shapes depend on the function.

    """
    def __init__(self, function, prefix=None):
        super(HybridLambda, self).__init__(prefix=prefix)
        if isinstance(function, str):
            assert hasattr(nd, function) and hasattr(sym, function), \
                   "Function name %s is not found in symbol/ndarray." % function
            func_dict = {sym: getattr(sym, function), nd: getattr(nd, function)}
            self._func = lambda F, *args: func_dict[F](*args)
            self._func_name = function
        elif callable(function):
            self._func = function
            self._func_name = function.__name__
        else:
            raise ValueError(
                "Unrecognized function in lambda: {} of type {}"
                .format(function, type(function)))

    def hybrid_forward(self, F, x, *args):
        return self._func(F, x, *args)

    def __repr__(self):
        return '{name}({function})'.format(name=self.__class__.__name__,
                                           function=self._func_name)
