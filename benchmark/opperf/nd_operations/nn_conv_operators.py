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

import mxnet as mx
from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

"""Performance benchmark tests for MXNet NDArray Convolution and Pooling Operators.

MXNet NDArray Pooling Operators

1. MaxPool1D
2. MaxPool2D
3. SumPool1D
4. SumPool2D
4. AvgPool1D
5. AvgPool2D
6. GlobalMaxPool1D
7. GlobalMaxPool2D
8. GlobalAvgPool1D
9. GlobalAvgPool2D
10.GlobalSumPool1D
11.GlobalSumPool2D
12.ROIPooling

(Under the hood uses mx.nd.pooling)

MXNet NDArray NN Convolution Operators

1. Conv1D
2. Conv2D
3. Conv1DTranspose (DeConvolution)
4. Conv2DTranspose (DeConvolution)

(Under the hood uses mx.nd.convolution, mx.nd.Deconvolution)

"""


def run_pooling_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and input data size (int64_tensor) for all the pooling
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    int64_tensor: str, default 'off'
        Input tensor size to use for tests (if on, dimensions >= 2**32)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    pool_types = ['avg', 'max', 'sum']
    global_pool_types = [0, 1]

    standard_data_list_pool1d = [(32, 3, 256), (32, 3, 64)]
    int64_tensor_data_list_pool1d = [(1, 1, 2**32)]
    standard_data_list_pool2d = [(32, 3, 256, 256), (32, 3, 64, 64)]
    int64_tensor_data_list_pool2d = [(2**28, 1, 4, 4)]
    standard_data_list_roipool = [(32, 3, 256, 256), (32, 3, 64, 64)]
    int64_tensor_data_list_roipool = [(32, 3, 2**13, 2**13)]

    if int64_tensor == 'on':
        data_list_pool1d = int64_tensor_data_list_pool1d
        data_list_pool2d = int64_tensor_data_list_pool2d
        data_list_roipool = int64_tensor_data_list_roipool
    else:
        data_list_pool1d = standard_data_list_pool1d
        data_list_pool2d = standard_data_list_pool2d
        data_list_roipool = standard_data_list_roipool

    # Run 1D and 2D Pooling performance runs
    pool1d_benchmark_res = []
    pool2d_benchmark_res = []
    for pool_type in pool_types:
        for global_pool in global_pool_types:
            for pool1d_data in data_list_pool1d:
                pool1d_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Pooling")],
                                                             run_backward=True,
                                                             dtype=dtype,
                                                             ctx=ctx,
                                                             profiler=profiler,
                                                             inputs=[{"data": pool1d_data,
                                                                      "kernel": 3,
                                                                      "pool_type": pool_type,
                                                                      "global_pool": global_pool,
                                                                      "stride": 1,
                                                                      "pad": 1}
                                                                    ],
                                                             warmup=warmup,
                                                             runs=runs)
            for pool2d_data in data_list_pool2d:
                pool2d_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Pooling")],
                                                             run_backward=True,
                                                             dtype=dtype,
                                                             ctx=ctx,
                                                             profiler=profiler,
                                                             inputs=[{"data": pool2d_data,
                                                                      "kernel": (3, 3),
                                                                      "pool_type": pool_type,
                                                                      "global_pool": global_pool,
                                                                      "stride": (1, 1),
                                                                      "pad": (0, 0)}
                                                                    ],
                                                             warmup=warmup,
                                                             runs=runs)
            # Run ROI Pooling performance runs
            roipool_benchmark_res = []
            for roipool_data in data_list_roipool:
                roipool_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "ROIPooling")],
                                                              run_backward=True,
                                                              dtype=dtype,
                                                              ctx=ctx,
                                                              profiler=profiler,
                                                              inputs=[{"data": roipool_data,
                                                                       "rois": (32, 5),
                                                                       "pooled_size": (2, 2),
                                                                       "spatial_scale": .5}
                                                                     ],
                                                              warmup=warmup,
                                                              runs=runs)
    # Prepare combined results
    mx_pooling_op_results = merge_map_list(pool1d_benchmark_res + pool2d_benchmark_res + roipool_benchmark_res)
    return mx_pooling_op_results


def run_convolution_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and input data size (int64_tensor) for all the convolution
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    int64_tensor: str, default 'off'
        Input tensor size to use for tests (if on, dimensions >= 2**32)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """

    standard_data_list_conv1d = [(32, 3, 256), (32, 3, 64)]
    int64_tensor_data_list_conv1d = [(2**30, 1, 4)]
    standard_weight_conv1d = (1, 3, 3)
    int64_tensor_weight_conv1d = (1, 1, 1)
    standard_kernel_conv1d = (3,)
    int64_tensor_kernel_conv1d = (1,)
    standard_data_list_conv2d = [(32, 3, 256, 256), (32, 3, 64, 64)]
    int64_tensor_data_list_conv2d = [(2**28, 1, 4, 4)]
    standard_weight_conv2d = (1, 3, 3, 3)
    int64_tensor_weight_conv2d = (1, 1, 1, 1)
    standard_kernel_conv2d = (3, 3)
    int64_tensor_kernel_conv2d = (1, 1)

    if int64_tensor == 'on':
        data_list_conv1d = int64_tensor_data_list_conv1d
        weight_conv1d = int64_tensor_weight_conv1d
        kernel_conv1d = int64_tensor_kernel_conv1d
        data_list_conv2d = int64_tensor_data_list_conv2d
        weight_conv2d = int64_tensor_weight_conv2d
        kernel_conv2d = int64_tensor_kernel_conv2d
    else:
        data_list_conv1d = standard_data_list_conv1d
        weight_conv1d = standard_weight_conv1d
        kernel_conv1d = standard_kernel_conv1d
        data_list_conv2d = standard_data_list_conv2d
        weight_conv2d = standard_weight_conv2d
        kernel_conv2d = standard_kernel_conv2d

    conv1d_benchmark_res = []
    conv2d_benchmark_res = []
    # Conv1D Benchmarks
    for conv_data in data_list_conv1d:
        conv1d_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Convolution")],
                                                     run_backward=True,
                                                     dtype=dtype,
                                                     ctx=ctx,
                                                     profiler=profiler,
                                                     inputs=[{"data": conv_data,
                                                              "weight": weight_conv1d,
                                                              "bias": (1,),
                                                              "kernel": kernel_conv1d,
                                                              "stride": (1,),
                                                              "dilate": (1,),
                                                              "pad": (0,),
                                                              "num_filter": 1,
                                                              "layout": 'NCW'}],
                                                     warmup=warmup,
                                                     runs=runs)
    # Conv2D Benchmarks
    for conv_data in data_list_conv2d:
        conv2d_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Convolution")],
                                                     run_backward=True,
                                                     dtype=dtype,
                                                     ctx=ctx,
                                                     profiler=profiler,
                                                     inputs=[{"data": conv_data,
                                                              "weight": weight_conv2d,
                                                              "bias": (1,),
                                                              "kernel": kernel_conv2d,
                                                              "stride": (1, 1),
                                                              "dilate": (1, 1),
                                                              "pad": (0, 0),
                                                              "num_filter": 1,
                                                              "layout": 'NCHW'}],
                                                     warmup=warmup,
                                                     runs=runs)
    # Prepare combined results
    mx_conv_op_results = merge_map_list(conv1d_benchmark_res + conv2d_benchmark_res)
    return mx_conv_op_results


def run_transpose_convolution_operators_benchmarks(ctx=mx.cpu(), profiler='native', int64_tensor='off', dtype='float32', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and input data size (int64_tensor) for all the transpose convolution
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    int64_tensor: str, default 'off'
        Input tensor size to use for tests (if on, dimensions >= 2**32)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """

    standard_data_list_conv1d_transpose = [(32, 3, 256), (32, 3, 64)]
    int64_tensor_data_list_conv1d_transpose = [(2**30, 1, 4)]
    standard_weight_conv1d_transpose = (3, 1, 3)
    int64_tensor_weight_conv1d_transpose = (1, 1, 1)
    standard_kernel_conv1d_transpose = (3,)
    int64_tensor_kernel_conv1d_transpose = (1,)
    standard_data_list_conv2d_transpose = [(32, 3, 256, 256), (32, 3, 64, 64)]
    int64_tensor_data_list_conv2d_transpose = [(2**28, 1, 4, 4)]
    standard_weight_conv2d_transpose = (3, 1, 3, 3)
    int64_tensor_weight_conv2d_transpose = (1, 1, 1, 1)
    standard_kernel_conv2d_transpose = (3, 3)
    int64_tensor_kernel_conv2d_transpose = (1, 1)

    if int64_tensor == 'on':
        data_list_conv1d_transpose = int64_tensor_data_list_conv1d_transpose
        weight_conv1d_transpose = int64_tensor_weight_conv1d_transpose
        kernel_conv1d_transpose = int64_tensor_kernel_conv1d_transpose
        data_list_conv2d_transpose = int64_tensor_data_list_conv2d_transpose
        weight_conv2d_transpose = int64_tensor_weight_conv2d_transpose
        kernel_conv2d_transpose = int64_tensor_kernel_conv2d_transpose
    else:
        data_list_conv1d_transpose = standard_data_list_conv1d_transpose
        weight_conv1d_transpose = standard_weight_conv1d_transpose
        kernel_conv1d_transpose = standard_kernel_conv1d_transpose
        data_list_conv2d_transpose = standard_data_list_conv2d_transpose
        weight_conv2d_transpose = standard_weight_conv2d_transpose
        kernel_conv2d_transpose = standard_kernel_conv2d_transpose

    # Conv1DTranspose Benchmarks
    conv1d_transpose_benchmark_res = []
    for conv_data in data_list_conv1d_transpose:
        conv1d_transpose_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Deconvolution")],
                                                                   run_backward=True,
                                                                   dtype=dtype,
                                                                   ctx=ctx,
                                                                   profiler=profiler,
                                                                   inputs=[{"data": conv_data,
                                                                            "weight": weight_conv1d_transpose,
                                                                            "bias": (1,),
                                                                            "kernel": kernel_conv1d_transpose,
                                                                            "stride": (1,),
                                                                            "dilate": (1,),
                                                                            "pad": (0,),
                                                                            "num_filter": 1,
                                                                            "no_bias": False,
                                                                            "layout": 'NCW'}],
                                                                   warmup=warmup,
                                                                   runs=runs)
    # Conv2DTranspose Benchmarks
    conv2d_transpose_benchmark_res = []
    for conv_data in data_list_conv2d_transpose:
        conv2d_transpose_benchmark_res += run_performance_test([getattr(MX_OP_MODULE, "Deconvolution")],
                                                                   run_backward=True,
                                                                   dtype=dtype,
                                                                   ctx=ctx,
                                                                   profiler=profiler,
                                                                   inputs=[{"data": conv_data,
                                                                            "weight": weight_conv2d_transpose,
                                                                            "bias": (1,),
                                                                            "kernel": kernel_conv2d_transpose,
                                                                            "stride": (1, 1),
                                                                            "pad": (0, 0),
                                                                            "num_filter": 1,
                                                                            "no_bias": False,
                                                                            "layout": 'NCHW'}],
                                                                   warmup=warmup,
                                                                   runs=runs)
    # Prepare combined results
    mx_transpose_conv_op_results = merge_map_list(conv1d_transpose_benchmark_res + conv2d_transpose_benchmark_res)
    return mx_transpose_conv_op_results
