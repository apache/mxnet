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
"""Custom convolutional neural network layers."""

__all__ = ['SpatialParallelConv2D', 'SpatialParallelConv3D',
           'SpatialParallelSplit', 'SpatialParallelAllgather']

from ...block import HybridBlock
from .... import symbol
from ....base import numeric_types, check_call, _LIB
from ...nn.activations import Activation
from ....util import is_np_array

def _infer_weight_shape(op_name, data_shape, kwargs):
    data = symbol.var('data', shape=data_shape)
    if is_np_array():
        op = getattr(symbol.npx, op_name)
        data = data.as_np_ndarray()
    else:
        op = getattr(symbol, op_name)
    sym = op(data, **kwargs)
    return sym.infer_shape_partial()[0]

class _SpatialParallelHelper(object):
    _init = False
    nccl_id = None
    num_gpus = None
    rank = None

    @staticmethod
    def init(num_gpus):
        """Communicate the NCCL unique id"""
        cls = _SpatialParallelHelper
        if not cls._init:
            cls._init = True
            import ctypes
            try:
                from mpi4py import MPI
            except:
                raise ImportError("Spatial parallel modules require mpi4py package.")
            import numpy as np
            nccl_id_size = ctypes.c_int()
            check_call(_LIB.MXNCCLGetUniqueIdSize(ctypes.byref(nccl_id_size)))
            nccl_id_size = nccl_id_size.value
            cls.nccl_id = np.zeros(nccl_id_size, np.byte)
            check_call(_LIB.MXNCCLGetUniqueId(
                cls.nccl_id.ctypes.data_as(ctypes.c_void_p)))
            global_comm = MPI.COMM_WORLD
            rank = global_comm.rank
            color = rank / num_gpus
            comm = global_comm.Split(color, rank)
            comm.Bcast([cls.nccl_id, nccl_id_size, MPI.BYTE], root=0)
            cls.num_gpus = num_gpus
            cls.rank = rank % num_gpus
        assert num_gpus == cls.num_gpus, ("All of the spatial parallel "
                                          "operations need to span the same number of GPUs")



class _SpatialParallelConv(HybridBlock):
    """Abstract nD spatial convolution layer (private, used as implementation base).
    It requires launching the script using MPI and single GPU per process.

    This layer creates a convolution kernel that is convolved
    with the layer input residing on multiple GPUs to produce a tensor of outputs.
    If `use_bias` is `True`, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    Parameters
    ----------
    num_gpus : int
        Specify the number of GPUs participating to compute the single convolution.
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NWC',
        'NHWC', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
        The spatial partitioning across different GPUs happens on the
        outermost dimension (e.g. 'H' in 2D case or 'D' in the 3D case).
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    cudnn_algo_fwd: int, default '-1'
        The cuDNN Convolution algorithm for forward computation.
        Value of '0' denotes the implicit GEMM algorithm. For
        more details refer to cuDNN documentation on cudnnConvolutionFwdAlgo_t.
    cudnn_algo_bwd_data: int, default '-1'
        The cuDNN Convolution algorithm for backward computation for data.
    cudnn_algo_bwd_filter: int, default '-1'
        The cuDNN Convolution algorithm for backward computation for filter.
    cudnn_tensor_core_only: bool, default False
        Whether to force cuDNN Convolution algorithm to use tensor cores.
    cudnn_algo_verbose : boolean, optional, default=False
        Verboseness of algo selection. 1 = output selection, 0 = no output
    cudnn_algo_fwd_prec : {'None', 'float16', 'float32', 'float64'}, optional, default='None'
        Precision of the computation of the forward convolution kernel.
        Default is the tensor data type, or float32 if the tensor data
        type is float16.
    cudnn_algo_bwd_prec : {'None', 'float16', 'float32', 'float64'}, optional, default='None'
        Precision of the computation of the back-prop kernels.
        Default is the tensor data type, or float32 if the tensor data
        type is float16.
    workspace: int, default 1024
         Maximum temporary workspace allowed (MB) in convolution.
         This parameter has two usages. When CUDNN is not used, it determines
         the effective batch size of the convolution kernel. When CUDNN is used,
         it controls the maximum temporary storage used for tuning the best CUDNN
         kernel when limited_workspace strategy is used. A large number leads to
         more (GPU) memory usage but may improve the performance.
    """
    def __init__(self, num_gpus, channels, kernel_size, strides, padding,
                 groups, layout, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 op_name='SpatialParallelConvolution', adj=None, prefix=None, params=None,
                 cudnn_algo_fwd=-1, cudnn_algo_bwd_data=-1, cudnn_algo_bwd_filter=-1,
                 cudnn_tensor_core_only=False, cudnn_algo_verbose=False,
                 cudnn_algo_fwd_prec='None', cudnn_algo_bwd_prec='None',
                 workspace=1024):
        super(_SpatialParallelConv, self).__init__(prefix=prefix, params=params)
        helper = _SpatialParallelHelper

        helper.init(num_gpus)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(strides, numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,)*len(kernel_size)
            assert kernel_size[0] % 2 != 0, \
                "Only supports odd values for the kernel_size[0] for now."
            assert padding[0] == int((kernel_size[0] - 1) / 2), \
                "Only supports padding[0] equal to the half of kernel_size[0] - 1 for now."
            self._op_name = op_name
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout,
                'num_gpus': num_gpus,
                'rank': helper.rank,
                'cudnn_algo_fwd': cudnn_algo_fwd,
                'cudnn_algo_bwd_data': cudnn_algo_bwd_data,
                'cudnn_algo_bwd_filter': cudnn_algo_bwd_filter,
                'cudnn_tensor_core_only': cudnn_tensor_core_only,
                'cudnn_algo_verbose': cudnn_algo_verbose,
                'cudnn_algo_fwd_prec': cudnn_algo_fwd_prec,
                'cudnn_algo_bwd_prec': cudnn_algo_bwd_prec,
                'workspace': workspace,
                'nccl_unique_id': helper.nccl_id.ctypes.data}
            if adj is not None:
                self._kwargs['adj'] = adj

            if is_np_array():
                dshape = [-1]*(len(kernel_size) + 2)
            else:
                dshape = [0]*(len(kernel_size) + 2)

            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels
            wshapes = _infer_weight_shape(op_name, dshape, self._kwargs)
            if kernel_size in [(1, 1), (1, 1, 1)]:
                self._spatial = False
                self._op_name = self._op_name.replace("SpatialParallel", "")
                self._kwargs.pop('num_gpus')
                self._kwargs.pop('nccl_unique_id')
                self._kwargs.pop('rank')
            else:
                self._spatial = True
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):
        if is_np_array():
            F = F.npx
        if bias is None:
            act = getattr(F, self._op_name)(x, weight=weight, name='fwd', **self._kwargs)
        else:
            act = getattr(F, self._op_name)(x, weight=weight, bias=bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'spatialparallelconv'

    def __repr__(self):
        s = '{name}({mapping}, num_gpus={num_gpus}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)

class SpatialParallelConv2D(_SpatialParallelConv):
    r"""2D convolution layer (e.g. spatial convolution over images),
    spatially distributed across multiple GPUs.

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    num_gpus : int
        Specify the number of GPUs participating to compute the single convolution.
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 2 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 2 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation : int or tuple/list of 2 int
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NHWC'
        Dimension ordering of data and weight. Only supports 'NHWC'
        layout for now. 'N', 'C', 'H', 'W' stands for batch, channel, height,
        and width dimensions respectively. Convolution is applied on the 'H' and
        'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, height, width, in_channels)` when `layout` is `NHWC`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, out_height, out_width, channels)` when `layout` is `NHWC`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    """
    def __init__(self, num_gpus, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 groups=1, layout='NHWC',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        assert layout in ('NHWC'), "Only supports 'NHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*2
        assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"
        op_name = kwargs.pop('op_name', 'SpatialParallelConvolution')
        if is_np_array():
            op_name = 'spatial_parallel_convolution'
        super(SpatialParallelConv2D, self).__init__(
            num_gpus, channels, kernel_size, strides, padding, groups, layout,
            in_channels, activation, use_bias, weight_initializer, bias_initializer,
            op_name, **kwargs)

class SpatialParallelConv3D(_SpatialParallelConv):
    """3D convolution layer (e.g. spatial convolution over volumes),
    spatially distributed across multiple GPUs.

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is `True`,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    If `in_channels` is not specified, `Parameter` initialization will be
    deferred to the first time `forward` is called and `in_channels` will be
    inferred from the shape of input data.

    Parameters
    ----------
    num_gpus : int
        Specify the number of GPUs participating to compute the single convolution.
    channels : int
        The dimensionality of the output space, i.e. the number of output
        channels (filters) in the convolution.
    kernel_size :int or tuple/list of 3 int
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 3 int,
        Specify the strides of the convolution.
    padding : int or a tuple/list of 3 int,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str, default 'NDHWC'
        Dimension ordering of data and weight. Only supports 'NDHWC'
        layout for now. 'N', 'C', 'H', 'W', 'D' stands for batch, channel, height,
        width and depth dimensions respectively. Convolution is applied on the 'D',
        'H' and 'W' dimensions. The spatial partitioning across different GPUs
        happens on the outermost dimension ('D').
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer : str or `Initializer`
        Initializer for the bias vector.


    Inputs:
        - **data**: 5D input tensor with shape
          `(batch_size, depth, height, width, in_channels)` when `layout` is `NDHWC`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 5D output tensor with shape
          `(batch_size, out_depth, out_height, out_width, channels)` when `layout` is `NDHWC`.
          out_depth, out_height and out_width are calculated as::

              out_depth = floor((depth+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_height = floor((height+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
              out_width = floor((width+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2])+1
    """
    def __init__(self, num_gpus, channels, kernel_size,
                 strides=(1, 1, 1), padding=(0, 0, 0),
                 groups=1, layout='NDHWC', activation=None,
                 use_bias=True, weight_initializer=None, bias_initializer='zeros',
                 in_channels=0, **kwargs):
        assert layout in ('NDHWC'), "Only supports 'NDHWC' layout for now"
        if isinstance(kernel_size, numeric_types):
            kernel_size = (kernel_size,)*3
        assert len(kernel_size) == 3, "kernel_size must be a number or a list of 3 ints"
        op_name = kwargs.pop('op_name', 'SpatialParallelConvolution')
        if is_np_array():
            op_name = 'spatial_parallel_convolution'
        super(SpatialParallelConv3D, self).__init__(num_gpus, channels, kernel_size, strides,
                                                    padding, groups, layout, in_channels,
                                                    activation, use_bias, weight_initializer,
                                                    bias_initializer, op_name, **kwargs)

class SpatialParallelSplit(HybridBlock):
    """Spatial parallel split"""
    def __init__(self, num_gpus, prefix=None, **kwargs):
        super(SpatialParallelSplit, self).__init__(prefix=prefix, **kwargs)
        helper = _SpatialParallelHelper
        helper.init(num_gpus)
        self._kwargs = {
            'num_gpus': num_gpus,
            'rank': helper.rank,
            'nccl_unique_id': helper.nccl_id.ctypes.data}

    def hybrid_forward(self, F, x):
        return F.contrib.SpatialParallelSplit(x, **self._kwargs)

class SpatialParallelAllgather(HybridBlock):
    """Spatial parallel allgather"""
    def __init__(self, num_gpus, prefix=None, **kwargs):
        super(SpatialParallelAllgather, self).__init__(prefix=prefix, **kwargs)
        helper = _SpatialParallelHelper
        helper.init(num_gpus)
        self._kwargs = {
            'num_gpus': num_gpus,
            'rank': helper.rank,
            'nccl_unique_id': helper.nccl_id.ctypes.data}

    def hybrid_forward(self, F, x):
        return F.contrib.SpatialParallelAllgather(x, **self._kwargs)
