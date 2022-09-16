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
           'BatchNorm', 'SyncBatchNorm', 'InstanceNorm', 'LayerNorm', 'GroupNorm',
           'Flatten', 'Lambda', 'HybridLambda', 'Concatenate', 'HybridConcatenate', 'Identity']
import warnings
import uuid
import numpy as _np

from .activations import Activation
from ..block import Block, HybridBlock
from ..utils import _indent
from ... import np, npx, device as _device
from ...util import use_np
from ..parameter import Parameter
from ...ndarray import get_dtype_name

class Sequential(Block):
    """Stacks Blocks sequentially.

    Example::

        net = nn.Sequential()
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(20))
    """
    def __init__(self):
        super(Sequential, self).__init__()
        self._layers = []

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self._layers.append(block)
            self.register_child(block)

    def forward(self, x, *args):
        for block in self._children.values():
            x = block()(x, *args)
            args = []
            if isinstance(x, (tuple, list)):
                args = x[1:]
                x = x[0]
        if args:
            x = tuple([x] + list(args))
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block().__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)()
            net.add(*(l() for l in layers))
            return net
        else:
            return layers()

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
        if self._children and all(isinstance(c(), HybridBlock) for c in self._children.values()):
            warnings.warn(
                f"All children of this Sequential layer '{repr(self)}'\n are HybridBlocks. Consider "
                "using HybridSequential for the best performance.", stacklevel=2)
        super(Sequential, self).hybridize(active, **kwargs)


@use_np
class HybridSequential(HybridBlock):
    """Stacks HybridBlocks sequentially.

    Example::

        net = nn.HybridSequential()
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(20))
        net.hybridize()
    """
    def __init__(self):
        super().__init__()
        self._layers = []

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self._layers.append(block)
            self.register_child(block)

    def forward(self, x, *args):
        for block in self._children.values():
            x = block()(x, *args)
            args = []
            if isinstance(x, (tuple, list)):
                args = x[1:]
                x = x[0]
        if args:
            x = tuple([x] + list(args))
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block().__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)()
            net.add(*(l() for l in layers))
            return net
        else:
            return layers()

    def __len__(self):
        return len(self._children)


@use_np
class Dense(HybridBlock):
    r"""Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, weight.T) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `weight` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

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
        self._units = units
        self._in_units = in_units
        self.weight = Parameter('weight', shape=(units, in_units),
                                init=weight_initializer, dtype=dtype,
                                allow_deferred_init=True)
        if use_bias:
            self.bias = Parameter('bias', shape=(units,),
                                  init=bias_initializer, dtype=dtype,
                                  allow_deferred_init=True)
        else:
            self.bias = None
        if activation is not None:
            self.act = Activation(activation)
        else:
            self.act = None

    def forward(self, x):
        device = x.device
        act = npx.fully_connected(x, self.weight.data(device),
                                  self.bias.data(device) if self.bias is not None else None,
                                  no_bias=self.bias is None,
                                  num_hidden=self._units, flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def infer_shape(self, x, *args):
        if self._flatten:
            num_input = 1
            for i in range(1, x.ndim):
                num_input *= x.shape[i]
            self.weight.shape = (self.weight.shape[0], num_input)
        else:
            self.weight.shape = (self.weight.shape[0], x.shape[x.ndim - 1])

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))


@use_np
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

    def forward(self, x):
        if self._rate > 0:
            return npx.dropout(x, p=self._rate, axes=self._axes, name='fwd', cudnn_off=False)
        else:
            return np.copy(x)

    def __repr__(self):
        s = '{name}(p = {_rate}, axes={_axes})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


@use_np
class _BatchNorm(HybridBlock):
    """Abstract BatchNorm layer (private, used as implementation base).
    Batch normalization layer (Ioffe and Szegedy, 2014).
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
                 use_global_stats=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, **kwargs):
        super(_BatchNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        self._axis = axis
        if in_channels != 0:
            self.in_channels = in_channels

        self.gamma = Parameter('gamma', grad_req='write' if scale else 'null',
                               shape=(in_channels,), init=gamma_initializer,
                               allow_deferred_init=True,
                               differentiable=scale)
        self.beta = Parameter('beta', grad_req='write' if center else 'null',
                              shape=(in_channels,), init=beta_initializer,
                              allow_deferred_init=True,
                              differentiable=center)
        self.running_mean = Parameter('running_mean', grad_req='null',
                                      shape=(in_channels,),
                                      init=running_mean_initializer,
                                      allow_deferred_init=True,
                                      differentiable=False)
        self.running_var = Parameter('running_var', grad_req='null',
                                     shape=(in_channels,),
                                     init=running_variance_initializer,
                                     allow_deferred_init=True,
                                     differentiable=False)

    def cast(self, dtype):
        if get_dtype_name(dtype) == 'float16':
            dtype = 'float32'
        super(_BatchNorm, self).cast(dtype)

    def forward(self, x):
        device = x.device
        return npx.batch_norm(x, self.gamma.data(device), self.beta.data(device),
                                  self.running_mean.data(device),
                                  self.running_var.data(device),
                                  name='fwd', **self._kwargs)

    def infer_shape(self, x, *args):
        channel_axis = self._axis if self._axis >= 0 else self._axis + x.ndim
        channel_count = x.shape[channel_axis]
        self.gamma.shape = (channel_count,)
        self.beta.shape = (channel_count,)
        self.running_mean.shape = (channel_count,)
        self.running_var.shape = (channel_count,)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))

class BatchNorm(_BatchNorm):
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
                 use_global_stats=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, **kwargs):
        super(BatchNorm, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale,
            use_global_stats=use_global_stats,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            running_mean_initializer=running_mean_initializer,
            running_variance_initializer=running_variance_initializer,
            in_channels=in_channels, **kwargs)


@use_np
class Embedding(HybridBlock):
    r"""Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]

    .. note::
        if `sparse_grad` is set to True, the gradient w.r.t weight will be
        sparse. Only a subset of optimizers support sparse gradients, including SGD,
        AdaGrad and Adam. By default lazy updates is turned on, which may perform
        differently from standard updates. For more details, please check the
        Optimization API at:
        https://mxnet.apache.org/versions/master/api/python/docs/api/optimizer/index.html

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
        assert not sparse_grad, "Currently, sparse feature is not supported in Gluon2.0"
        grad_stype = 'row_sparse' if sparse_grad else 'default'
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype, 'sparse_grad': sparse_grad}
        self.weight = Parameter('weight', shape=(input_dim, output_dim),
                                init=weight_initializer, dtype=dtype,
                                allow_deferred_init=True, grad_stype=grad_stype)

    def forward(self, x):
        device = x.device
        return npx.embedding(x, self.weight.data(device), name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__,
                        **self._kwargs)


@use_np
class Flatten(HybridBlock):
    r"""Flattens the input to two dimensional.

    Inputs:
        - **data**: input tensor with arbitrary shape `(N, x1, x2, ..., xn)`

    Output:
        - **out**: 2D tensor with shape: `(N, x1 \cdot x2 \cdot ... \cdot xn)`
    """
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def forward(self, x):
        return npx.batch_flatten(x)

    def __repr__(self):
        return self.__class__.__name__


@use_np
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
    >>> x = mx.np.array([[[ 1.1,  2.2]],
    ...                 [[ 3.3,  4.4]]])
    >>> # Instance normalization is calculated with the above formula
    >>> layer = InstanceNorm()
    >>> layer.initialize(device=mx.cpu(0))
    >>> layer(x)
    [[[-0.99998355  0.99998331]]
     [[-0.99998319  0.99998361]]]
    """
    def __init__(self, axis=1, epsilon=1e-5, center=True, scale=False,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self._kwargs = {'eps': epsilon, 'axis': axis, 'center': center, 'scale': scale}
        self._axis = axis
        self._epsilon = epsilon
        self.gamma = Parameter('gamma', grad_req='write' if scale else 'null',
                               shape=(in_channels,), init=gamma_initializer,
                               allow_deferred_init=True)
        self.beta = Parameter('beta', grad_req='write' if center else 'null',
                              shape=(in_channels,), init=beta_initializer,
                              allow_deferred_init=True)

    def forward(self, x):
        device = x.device
        if self._axis == 1:
            return npx.instance_norm(x, self.gamma.data(device), self.beta.data(device),
                                     name='fwd', eps=self._epsilon)
        x = x.swapaxes(1, self._axis)
        return npx.instance_norm(x, self.gamma.data(device), self.beta.data(device),
                                 name='fwd', eps=self._epsilon).swapaxes(1, self._axis)

    def infer_shape(self, x, *args):
        self.gamma.shape = (x.shape[1],)
        self.beta.shape = (x.shape[1],)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


@use_np
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
    >>> x = mx.np.array([[1, 2, 3, 4, 5], [1, 1, 2, 2, 2]])
    >>> # Layer normalization is calculated with the above formula
    >>> layer = LayerNorm()
    >>> layer.initialize(device=mx.cpu(0))
    >>> layer(x)
    [[-1.41421    -0.707105    0.          0.707105    1.41421   ]
     [-1.2247195  -1.2247195   0.81647956  0.81647956  0.81647956]]
    """
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0):
        super(LayerNorm, self).__init__()
        self._kwargs = {'eps': epsilon, 'axis': axis, 'center': center, 'scale': scale}
        self._axis = axis
        self._epsilon = epsilon
        self._center = center
        self._scale = scale
        self.gamma = Parameter('gamma', grad_req='write' if scale else 'null',
                               shape=(in_channels,), init=gamma_initializer,
                               allow_deferred_init=True)
        self.beta = Parameter('beta', grad_req='write' if center else 'null',
                              shape=(in_channels,), init=beta_initializer,
                              allow_deferred_init=True)

    def forward(self, data):
        device = data.device
        return npx.layer_norm(data, gamma=self.gamma.data(device),
                              beta=self.beta.data(device), axis=self._axis, eps=self._epsilon)

    def infer_shape(self, data, *args):
        channel_axis = self._axis if self._axis >= 0 else self._axis + data.ndim
        channel_count = data.shape[channel_axis]
        self.gamma.shape = (channel_count,)
        self.beta.shape = (channel_count,)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


@use_np
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
    >>> x = mx.np.array([[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]])
    >>> # Group normalization is calculated with the above formula
    >>> layer = GroupNorm()
    >>> layer.initialize(device=mx.cpu(0))
    >>> layer(x)
    [[[-1.5932543 -1.3035717 -1.0138891 -0.7242065]
      [-0.4345239 -0.1448413  0.1448413  0.4345239]
      [ 0.7242065  1.0138891  1.3035717  1.5932543]]
     [[-1.5932543 -1.3035717 -1.0138891 -0.7242065]
      [-0.4345239 -0.1448413  0.1448413  0.4345239]
      [ 0.7242065  1.0138891  1.3035717  1.5932543]]]
    """
    def __init__(self, num_groups=1, epsilon=1e-5, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 in_channels=0):
        super(GroupNorm, self).__init__()
        self._kwargs = {'eps': epsilon, 'num_groups': num_groups, 'center': center, 'scale': scale}
        self._num_groups = num_groups
        self._epsilon = epsilon
        self._center = center
        self._scale = scale
        self.gamma = Parameter('gamma', grad_req='write' if scale else 'null',
                               shape=(in_channels,), init=gamma_initializer,
                               allow_deferred_init=True)
        self.beta = Parameter('beta', grad_req='write' if center else 'null',
                              shape=(in_channels,), init=beta_initializer,
                              allow_deferred_init=True)

    def forward(self, data):
        device = data.device
        norm_data = npx.group_norm(data, gamma=self.gamma.data(device), beta=self.beta.data(device),
                                   num_groups=self._num_groups, eps=self._epsilon)
        return norm_data

    def infer_shape(self, data, *args):
        self.gamma.shape = (data.shape[1],)
        self.beta.shape = (data.shape[1],)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels)
        s += ')'
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

            block = Lambda(lambda x: npx.leaky_relu(x, slope=0.1))

    Inputs:
        - ** *args **: one or more input data. Their shapes depend on the function.

    Output:
        - ** *outputs **: one or more output data. Their shapes depend on the function.
    """
    def __init__(self, function):
        super(Lambda, self).__init__()
        if isinstance(function, str):
            if hasattr(np, function):
                self._func_impl = getattr(np, function)
            elif hasattr(npx, function):
                self._func_impl = getattr(npx, function)
            else:
                raise Exception(f"Function name {function} is not found in np/npx.")
            self._func_name = function
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


@use_np
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
    def __init__(self, function):
        super(HybridLambda, self).__init__()
        if isinstance(function, str):
            if hasattr(np, function):
                self._func = getattr(np, function)
            elif hasattr(npx, function):
                self._func = getattr(npx, function)
            else:
                raise Exception(f"Function name {function} is not found in np/npx.")
            self._func_name = function
        elif callable(function):
            self._func = function
            self._func_name = function.__name__
        else:
            raise ValueError(
                "Unrecognized function in lambda: {} of type {}"
                .format(function, type(function)))

    def forward(self, x, *args):
        return self._func(x, *args)

    def __repr__(self):
        return '{name}({function})'.format(name=self.__class__.__name__,
                                           function=self._func_name)


@use_np
class Concatenate(Sequential):
    """Lays `Block` s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = Concatenate()
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(20))
        net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1):
        super(Concatenate, self).__init__()
        self.axis = axis

    def forward(self, x):
        out = []
        for block in self._children.values():
            out.append(block()(x))
        out = np.concatenate(out, axis=self.axis)
        return out


@use_np
class HybridConcatenate(HybridSequential):
    """Lays `HybridBlock` s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = HybridConcatenate()
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(20))
        net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        out = []
        for block in self._children.values():
            out.append(block()(x))
        out = np.concatenate(out, axis=self.axis)
        return out


@use_np
class Identity(HybridBlock):
    """Block that passes through the input directly.

    This block can be used in conjunction with HybridConcatenate
    block for residual connection.

    Example::

        net = HybridConcatenate()
        net.add(nn.Dense(10, activation='relu'))
        net.add(nn.Dense(20))
        net.add(Identity())
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


@use_np
class SyncBatchNorm(BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)

    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the implementation described in the paper [2]_.

    Note: Current implementation of SyncBN does not support FP16 training.
    For FP16 inference, use standard nn.BatchNorm instead of SyncBN.

    Parameters
    ----------
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    num_devices : int, default number of visible GPUs
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


    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.

    Reference:
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating \
          deep network training by reducing internal covariate shift." *ICML 2015*
        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, \
          Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    """
    def __init__(self, in_channels=0, num_devices=None, momentum=0.9, epsilon=1e-5,
                 center=True, scale=True, use_global_stats=False, beta_initializer='zeros',
                 gamma_initializer='ones', running_mean_initializer='zeros',
                 running_variance_initializer='ones', **kwargs):
        super(SyncBatchNorm, self).__init__(
            axis=1, momentum=momentum, epsilon=epsilon,
            center=center, scale=scale,
            use_global_stats=use_global_stats,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            running_mean_initializer=running_mean_initializer,
            running_variance_initializer=running_variance_initializer,
            in_channels=in_channels, **kwargs)
        num_devices = self._get_num_devices() if num_devices is None else num_devices
        self._kwargs = {'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats,
                        'ndev': num_devices, 'key': uuid.uuid4()}

    def _get_num_devices(self):
        warnings.warn("Caution using SyncBatchNorm: "
                      "if not using all the GPUs, please mannually set num_devices",
                      UserWarning)
        num_devices = _device.num_gpus()
        num_devices = num_devices if num_devices > 0 else 1
        return num_devices

    def forward(self, x):
        device = x.device
        return npx.sync_batch_norm(x, self.gamma.data(device), self.beta.data(device),
                                   self.running_mean.data(device), self.running_var.data(device),
                                   name='fwd', **self._kwargs)
