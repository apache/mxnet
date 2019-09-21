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
"""Custom neural network layers in model_zoo."""

__all__ = ['Concurrent', 'HybridConcurrent', 'Identity', 'SparseEmbedding',
           'SyncBatchNorm', 'PixelShuffle1D', 'PixelShuffle2D',
           'PixelShuffle3D']

import warnings
from .... import nd, context
from ...block import HybridBlock, Block
from ...nn import Sequential, HybridSequential, BatchNorm

class Concurrent(Sequential):
    """Lays `Block` s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = Concurrent()
        # use net's name_scope to give children blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1, prefix=None, params=None):
        super(Concurrent, self).__init__(prefix=prefix, params=params)
        self.axis = axis

    def forward(self, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        out = nd.concat(*out, dim=self.axis)
        return out


class HybridConcurrent(HybridSequential):
    """Lays `HybridBlock` s concurrently.

    This block feeds its input to all children blocks, and
    produce the output by concatenating all the children blocks' outputs
    on the specified axis.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give children blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())

    Parameters
    ----------
    axis : int, default -1
        The axis on which to concatenate the outputs.
    """
    def __init__(self, axis=-1, prefix=None, params=None):
        super(HybridConcurrent, self).__init__(prefix=prefix, params=params)
        self.axis = axis

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        out = F.concat(*out, dim=self.axis)
        return out


class Identity(HybridBlock):
    """Block that passes through the input directly.

    This block can be used in conjunction with HybridConcurrent
    block for residual connection.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())
    """
    def __init__(self, prefix=None, params=None):
        super(Identity, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        return x

class SparseEmbedding(Block):
    r"""Turns non-negative integers (indexes/tokens) into dense vectors
    of fixed size. eg. [4, 20] -> [[0.25, 0.1], [0.6, -0.2]]

    This SparseBlock is designed for distributed training with extremely large
    input dimension. Both weight and gradient w.r.t. weight are `RowSparseNDArray`.

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

    Inputs:
        - **data**: (N-1)-D tensor with shape: `(x1, x2, ..., xN-1)`.
    Output:
        - **out**: N-D tensor with shape: `(x1, x2, ..., xN-1, output_dim)`.
    """
    def __init__(self, input_dim, output_dim, dtype='float32',
                 weight_initializer=None, **kwargs):
        super(SparseEmbedding, self).__init__(**kwargs)
        self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim,
                        'dtype': dtype, 'sparse_grad': True}
        self.weight = self.params.get('weight', shape=(input_dim, output_dim),
                                      init=weight_initializer, dtype=dtype,
                                      grad_stype='row_sparse', stype='row_sparse')

    def forward(self, x):
        weight = self.weight.row_sparse_data(x)
        return nd.Embedding(x, weight, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__,
                        **self._kwargs)

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
        super(SyncBatchNorm, self).__init__(1, momentum, epsilon, center, scale, use_global_stats,
                                            beta_initializer, gamma_initializer,
                                            running_mean_initializer, running_variance_initializer,
                                            in_channels, **kwargs)
        num_devices = self._get_num_devices() if num_devices is None else num_devices
        self._kwargs = {'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats,
                        'ndev': num_devices, 'key': self.prefix}

    def _get_num_devices(self):
        warnings.warn("Caution using SyncBatchNorm: "
                      "if not using all the GPUs, please mannually set num_devices",
                      UserWarning)
        num_devices = context.num_gpus()
        num_devices = num_devices if num_devices > 0 else 1
        return num_devices

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.contrib.SyncBatchNorm(x, gamma, beta, running_mean, running_var,
                                       name='fwd', **self._kwargs)

class PixelShuffle1D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 1 dimension.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``W`` dimension, thereby effectively multiplying that dimension
    by a constant factor in size.

    For example, a feature map of shape :math:`(fC, W)` is reshaped
    into :math:`(C, fW)` by forming little value groups of size :math:`f`
    and arranging them in a grid of size :math:`W`.

    Parameters
    ----------
    factor : int or 1-tuple of int
        Upsampling factor, applied to the ``W`` dimension.

    Inputs:
        - **data**: Tensor of shape ``(N, f*C, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, W*f)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle1D(2)
    >>> x = mx.nd.zeros((1, 8, 3))
    >>> pxshuf(x).shape
    (1, 4, 6)
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self._factor = int(factor)

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        f = self._factor
                                             # (N, C*f, W)
        x = F.reshape(x, (0, -4, -1, f, 0))  # (N, C, f, W)
        x = F.transpose(x, (0, 1, 3, 2))     # (N, C, W, f)
        x = F.reshape(x, (0, 0, -3))         # (N, C, W*f)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factor)


class PixelShuffle2D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 2 dimensions.

    Pixel-shuffling is the operation of taking groups of values along
    the *channel* dimension and regrouping them into blocks of pixels
    along the ``H`` and ``W`` dimensions, thereby effectively multiplying
    those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^2 C, H, W)` is reshaped
    into :math:`(C, fH, fW)` by forming little :math:`f \times f` blocks
    of pixels and arranging them in an :math:`H \times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 2-tuple of int
        Upsampling factors, applied to the ``H`` and ``W`` dimensions,
        in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*C, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, H*f1, W*f2)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle2D((2, 3))
    >>> x = mx.nd.zeros((1, 12, 3, 5))
    >>> pxshuf(x).shape
    (1, 2, 6, 15)
    """

    def __init__(self, factor):
        super(PixelShuffle2D, self).__init__()
        try:
            self._factors = (int(factor),) * 2
        except TypeError:
            self._factors = tuple(int(fac) for fac in factor)
            assert len(self._factors) == 2, "wrong length {}".format(len(self._factors))

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        f1, f2 = self._factors
                                                      # (N, f1*f2*C, H, W)
        x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C, f1*f2, H, W)
        x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))    # (N, C, f1, f2, H, W)
        x = F.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
        x = F.reshape(x, (0, 0, -3, -3))              # (N, C, H*f1, W*f2)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factors)


class PixelShuffle3D(HybridBlock):

    r"""Pixel-shuffle layer for upsampling in 3 dimensions.

    Pixel-shuffling (or voxel-shuffling in 3D) is the operation of taking
    groups of values along the *channel* dimension and regrouping them into
    blocks of voxels along the ``D``, ``H`` and ``W`` dimensions, thereby
    effectively multiplying those dimensions by a constant factor in size.

    For example, a feature map of shape :math:`(f^3 C, D, H, W)` is reshaped
    into :math:`(C, fD, fH, fW)` by forming little :math:`f \times f \times f`
    blocks of voxels and arranging them in a :math:`D \times H \times W` grid.

    Pixel-shuffling together with regular convolution is an alternative,
    learnable way of upsampling an image by arbitrary factors. It is reported
    to help overcome checkerboard artifacts that are common in upsampling with
    transposed convolutions (also called deconvolutions). See the paper
    `Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
    for further details.

    Parameters
    ----------
    factor : int or 3-tuple of int
        Upsampling factors, applied to the ``D``, ``H`` and ``W``
        dimensions, in that order.

    Inputs:
        - **data**: Tensor of shape ``(N, f1*f2*f3*C, D, H, W)``.
    Outputs:
        - **out**: Tensor of shape ``(N, C, D*f1, H*f2, W*f3)``.

    Examples
    --------
    >>> pxshuf = PixelShuffle3D((2, 3, 4))
    >>> x = mx.nd.zeros((1, 48, 3, 5, 7))
    >>> pxshuf(x).shape
    (1, 2, 6, 15, 28)
    """

    def __init__(self, factor):
        super(PixelShuffle3D, self).__init__()
        try:
            self._factors = (int(factor),) * 3
        except TypeError:
            self._factors = tuple(int(fac) for fac in factor)
            assert len(self._factors) == 3, "wrong length {}".format(len(self._factors))

    def hybrid_forward(self, F, x):
        """Perform pixel-shuffling on the input."""
        # `transpose` doesn't support 8D, need other implementation
        f1, f2, f3 = self._factors
                                                              # (N, C*f1*f2*f3, D, H, W)
        x = F.reshape(x, (0, -4, -1, f1 * f2 * f3, 0, 0, 0))  # (N, C, f1*f2*f3, D, H, W)
        x = F.swapaxes(x, 2, 3)                               # (N, C, D, f1*f2*f3, H, W)
        x = F.reshape(x, (0, 0, 0, -4, f1, f2*f3, 0, 0))      # (N, C, D, f1, f2*f3, H, W)
        x = F.reshape(x, (0, 0, -3, 0, 0, 0))                 # (N, C, D*f1, f2*f3, H, W)
        x = F.swapaxes(x, 3, 4)                               # (N, C, D*f1, H, f2*f3, W)
        x = F.reshape(x, (0, 0, 0, 0, -4, f2, f3, 0))         # (N, C, D*f1, H, f2, f3, W)
        x = F.reshape(x, (0, 0, 0, -3, 0, 0))                 # (N, C, D*f1, H*f2, f3, W)
        x = F.swapaxes(x, 4, 5)                               # (N, C, D*f1, H*f2, W, f3)
        x = F.reshape(x, (0, 0, 0, 0, -3))                    # (N, C, D*f1, H*f2, W*f3)
        return x

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._factors)
