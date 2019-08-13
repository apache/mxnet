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
from mxnet import nd
from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list

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

(Under the hood uses mx.nd.pooling)

MXNet NDArray NN Convolution Operators

1. Conv1D
2. Conv2D
3. Conv1DTranspose (DeConvolution)
4. Conv2DTranspose (DeConvolution)

(Under the hood uses mx.nd.convolution, mx.nd.Deconvolution)

"""


def run_pooling_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    pool_types = ['avg', 'max', 'sum']
    global_pool_types = [0, 1]

    # Run 1D and 2D Pooling performance runs
    pool1d_benchmark_res = []
    pool2d_benchmark_res = []
    for pool_type in pool_types:
        for global_pool in global_pool_types:
            for pool1d_data in [(32, 3, 256), (32, 3, 64)]:
                pool1d_benchmark_res += run_performance_test([nd.Pooling],
                                                             run_backward=True,
                                                             dtype=dtype,
                                                             ctx=ctx,
                                                             inputs=[{"data": pool1d_data,
                                                                      "kernel": 3,
                                                                      "pool_type": pool_type,
                                                                      "global_pool": global_pool,
                                                                      "stride": 1,
                                                                      "pad": 1,
                                                                      "layout": 'NCW'}
                                                                     ],
                                                             warmup=warmup,
                                                             runs=runs)
            for pool2d_data in [(32, 3, 256, 256), (32, 3, 64, 64)]:
                pool2d_benchmark_res += run_performance_test([nd.Pooling],
                                                             run_backward=True,
                                                             dtype=dtype,
                                                             ctx=ctx,
                                                             inputs=[{"data": pool2d_data,
                                                                      "kernel": (3, 3),
                                                                      "pool_type": pool_type,
                                                                      "global_pool": global_pool,
                                                                      "stride": (1, 1),
                                                                      "pad": (0, 0),
                                                                      "layout": 'NCHW'}
                                                                     ],
                                                             warmup=warmup,
                                                             runs=runs)
    # Prepare combined results
    mx_pooling_op_results = merge_map_list(pool1d_benchmark_res + pool2d_benchmark_res)
    return mx_pooling_op_results


def run_convolution_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    # Conv1D Benchmarks
    conv1d_benchmark_res = []
    for conv_data in [(32, 3, 256), (32, 3, 64)]:
        conv1d_benchmark_res += run_performance_test([nd.Convolution],
                                                     run_backward=True,
                                                     dtype=dtype,
                                                     ctx=ctx,
                                                     inputs=[{"data": conv_data,
                                                              "weight": (64, 3, 3,),
                                                              "bias": (64,),
                                                              "kernel": (3,),
                                                              "stride": (1,),
                                                              "dilate": (1,),
                                                              "pad": (0,),
                                                              "num_filter": 64,
                                                              "layout": 'NCW'}
                                                             ],
                                                     warmup=warmup,
                                                     runs=runs)
    # Conv2D Benchmarks
    conv2d_benchmark_res = []
    for conv_data in [(32, 3, 256, 256), (32, 3, 64, 64)]:
        conv2d_benchmark_res += run_performance_test([nd.Convolution],
                                                     run_backward=True,
                                                     dtype=dtype,
                                                     ctx=ctx,
                                                     inputs=[{"data": conv_data,
                                                              "weight": (64, 3, 3, 3),
                                                              "bias": (64,),
                                                              "kernel": (3, 3),
                                                              "stride": (1, 1),
                                                              "dilate": (1, 1),
                                                              "pad": (0, 0),
                                                              "num_filter": 64,
                                                              "layout": 'NCHW'}
                                                             ],
                                                     warmup=warmup,
                                                     runs=runs)
    # Prepare combined results
    mx_conv_op_results = merge_map_list(conv1d_benchmark_res + conv2d_benchmark_res)
    return mx_conv_op_results
