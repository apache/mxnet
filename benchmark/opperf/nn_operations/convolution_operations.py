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

from mxnet.gluon import nn

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list

""" Performance benchmark tests for MXNet Gluon Convolution Layers

1. Conv2D

TODO

2. Conv1D
3. Conv1DTranspose
4. Conv2DTranspose

NOTE: Number of warmup and benchmark runs for convolution may need to be reduced as the computation
is heavy and within first 25 runs results stabilizes without variation.
"""


def run_convolution_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=25):
    """Runs benchmarks with the given context and precision (dtype)for all convolution Gluon blocks
    in MXNet.

    :param ctx: Context to run benchmarks
    :param dtype: Precision to use for benchmarks
    :param warmup: Number of times to run for warmup
    :param runs: Number of runs to capture benchmark results
    :return: Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """

    # Benchmark Gluon Conv2D Block.
    conv2d_res = run_performance_test(nn.Conv2D, run_backward=True, dtype=dtype, ctx=ctx,
                                      inputs=[{"data": (32, 3, 256, 256),
                                               "channels": 64,
                                               "kernel_size": (3, 3),
                                               "strides": (1, 1),
                                               "padding": (0, 0),
                                               "dilation": (1, 1),
                                               "layout": "NCHW",
                                               "activation": None, }],
                                      warmup=warmup, runs=runs)

    # Prepare combined results for Gluon Convolution operators
    mx_gluon_conv_op_results = merge_map_list([conv2d_res])
    return mx_gluon_conv_op_results
