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
           'SyncBatchNorm']

import warnings
from .... import nd, test_utils
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
    We follow the sync-onece implmentation described in the paper [2]_.

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
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.


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
        num_devices = len(test_utils.list_gpus())
        num_devices = num_devices if num_devices > 0 else 1
        return num_devices

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.contrib.SyncBatchNorm(x, gamma, beta, running_mean, running_var,
                                       name='fwd', **self._kwargs)
