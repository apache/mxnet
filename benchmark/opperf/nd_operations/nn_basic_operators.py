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

"""Performance benchmark tests for MXNet NDArray basic NN Operators.

1. FullyConnected
2. Dropout
3. BatchNorm

"""


def run_nn_basic_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    # FullyConnnected operator benchmarks
    fc_benchmark_res = run_performance_test([nd.FullyConnected],
                                            run_backward=True,
                                            dtype=dtype,
                                            ctx=ctx,
                                            inputs=[{"data": (32, 3, 256, 256),
                                                     "num_hidden": 64,
                                                     "weight": (64, 3 * 256 * 256),
                                                     "bias": (64,),
                                                     "flatten": True},
                                                    {"data": (32, 3, 256, 256),
                                                     "num_hidden": 64,
                                                     "weight": (64, 256),
                                                     "bias": (64,),
                                                     "flatten": False}],
                                            warmup=warmup,
                                            runs=runs)

    # Dropout benchmarks
    dropout_benchmark_res = run_performance_test([nd.Dropout],
                                                 run_backward=True,
                                                 dtype=dtype,
                                                 ctx=ctx,
                                                 inputs=[{"data": (32, 3, 256, 256),
                                                          "p": 0.5,
                                                          "mode": "always"},
                                                         {"data": (10000, 10),
                                                          "p": 0.5,
                                                          "mode": "always"}],
                                                 warmup=warmup,
                                                 runs=runs)
    # BatchNorm benchmarks
    batchnorm_benchmark_res = run_performance_test([nd.BatchNorm],
                                                   run_backward=True,
                                                   dtype=dtype,
                                                   ctx=ctx,
                                                   inputs=[{"data": (32, 3, 256, 256),
                                                            "gamma": (3,),
                                                            "beta": (3,),
                                                            "moving_mean": (3,),
                                                            "moving_var": (3,)},
                                                           {"data": (32, 3, 10000, 10),
                                                            "gamma": (3,),
                                                            "beta": (3,),
                                                            "moving_mean": (3,),
                                                            "moving_var": (3,)}],
                                                   warmup=warmup,
                                                   runs=runs)
    # Prepare combined results
    mx_basic_nn_results = merge_map_list(fc_benchmark_res + dropout_benchmark_res + batchnorm_benchmark_res)
    return mx_basic_nn_results
