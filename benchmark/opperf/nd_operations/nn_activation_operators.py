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

"""Performance benchmark tests for MXNet NDArray Activation Operators.

1. LeakyRelu
    1.1 Elu
    1.2 Selu
    1.3 Leaky
    1.4 PRelu
    1.5 RRelu
3. Hard_Sigmoid
4. Softmax
5. Log_Softmax

"""


def run_activation_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the activation
    operators (relu, sigmoid, softmax) in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    warmup: int, default 10
        Number of times to run for warmup
    runs: int, default 50
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Relu and its variation
    relu_benchmark_res = run_performance_test([nd.LeakyReLU],
                                              run_backward=True,
                                              dtype=dtype,
                                              ctx=ctx,
                                              inputs=[{"data": (1024, 1024), "act_type": "leaky", "slope": 0.1},
                                                      {"data": (10000, 1), "act_type": "leaky", "slope": 0.1},
                                                      {"data": (10000, 100), "act_type": "leaky", "slope": 0.1},
                                                      {"data": (1024, 1024), "act_type": "elu", "slope": 0.1},
                                                      {"data": (10000, 1), "act_type": "elu", "slope": 0.1},
                                                      {"data": (10000, 100), "act_type": "elu", "slope": 0.1},
                                                      {"data": (1024, 1024), "act_type": "selu"},
                                                      {"data": (10000, 1), "act_type": "selu"},
                                                      {"data": (10000, 100), "act_type": "selu"},
                                                      {"data": (1024, 1024), "act_type": "prelu", "gamma": (1, 1024)},
                                                      {"data": (10000, 1), "act_type": "prelu", "gamma": (1, 1)},
                                                      {"data": (10000, 100), "act_type": "prelu", "gamma": (1, 100)}
                                                      ],
                                              warmup=warmup,
                                              runs=runs)

    # Sigmoid => Covered as part of Unary ops
    # Hard_Sigmoid
    hard_sigmoid_benchmark_res = run_performance_test([nd.hard_sigmoid],
                                                      run_backward=True,
                                                      dtype=dtype,
                                                      ctx=ctx,
                                                      inputs=[{"data": (1024, 1024), "alpha": 0.25, "beta": 0.5},
                                                              {"data": (10000, 1), "alpha": 0.25, "beta": 0.5},
                                                              {"data": (10000, 100), "alpha": 0.25, "beta": 0.5}
                                                              ],
                                                      warmup=warmup,
                                                      runs=runs)

    # Softmax, LogSoftmax
    softmax_benchmark_res = run_performance_test([nd.softmax, nd.log_softmax],
                                                 run_backward=True,
                                                 dtype=dtype,
                                                 ctx=ctx,
                                                 inputs=[{"data": (1024, 1024), "axis": -1, "temperature": 0.5},
                                                         {"data": (10000, 1), "axis": -1, "temperature": 0.5},
                                                         {"data": (10000, 100), "axis": -1, "temperature": 0.5}
                                                         ],
                                                 warmup=warmup,
                                                 runs=runs)

    # Prepare combined results
    mx_activation_op_results = merge_map_list(relu_benchmark_res + hard_sigmoid_benchmark_res + softmax_benchmark_res)
    return mx_activation_op_results
