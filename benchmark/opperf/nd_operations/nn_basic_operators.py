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

from benchmark.opperf.utils.op_registry_utils import get_all_nn_basic_operators
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

"""Performance benchmark tests for MXNet NDArray basic NN Operators.

1. FullyConnected
2. Dropout
3. BatchNorm
4. SoftmaxOutput
5. LinearRegressionOutput
6. LogisticRegressionOutput
7. MAERegressionOutput
8. SVMOutput
9. L2Normalization
10. LayerNorm
11. InstanceNorm
12. Embedding
13. Correlation
14. SpatialTransformer
15. im2col
16. col2im
17. GroupNorm
18. RNN
19. LRN

"""


def run_nn_basic_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and data size (int64_tensor) for all the basic neural network
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

    standard_data_list = [(1024, 4, 4)]
    int64_tensor_data_list = [(2**28, 4, 4)]

    if int64_tensor == 'on':
        data_list = int64_tensor_data_list
    else:
        data_list = standard_data_list

    for data in data_list:
        rnn_relu_benchmark = run_performance_test([getattr(MX_OP_MODULE, "RNN")],
                                                  run_backward=True,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  profiler=profiler,
                                                  inputs=[{"data": data,
                                                           "parameters": (7,),
                                                           "state": (1, 4, 1),
                                                           "mode": "rnn_relu",
                                                           "state_size": 1,
                                                           "num_layers": 1}],
                                                  warmup=warmup,
                                                  runs=runs)
        rnn_tanh_benchmark = run_performance_test([getattr(MX_OP_MODULE, "RNN")],
                                                  run_backward=True,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  profiler=profiler,
                                                  inputs=[{"data": data,
                                                           "parameters": (7,),
                                                           "state": (1, 4, 1),
                                                           "mode": "rnn_tanh",
                                                           "state_size": 1,
                                                           "num_layers": 1}],
                                                  warmup=warmup,
                                                  runs=runs)
        rnn_lstm_benchmark = run_performance_test([getattr(MX_OP_MODULE, "RNN")],
                                                  run_backward=True,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  profiler=profiler,
                                                  inputs=[{"data": data,
                                                           "parameters": (28,),
                                                           "state": (1, 4, 1),
                                                           "state_cell": (1, 4, 1),
                                                           "mode": "lstm",
                                                           "state_size": 1,
                                                           "num_layers": 1}],
                                                  warmup=warmup,
                                                  runs=runs)
        rnn_gru_benchmark = run_performance_test([getattr(MX_OP_MODULE, "RNN")],
                                                 run_backward=True,
                                                 dtype=dtype,
                                                 ctx=ctx,
                                                 profiler=profiler,
                                                 inputs=[{"data": data,
                                                          "parameters": (21,),
                                                          "state": (1, 4, 1),
                                                          "mode": "gru",
                                                          "state_size": 1,
                                                          "num_layers": 1}],
                                                 warmup=warmup,
                                                 runs=runs)
    # Fetch all NN Basic Operators
    mx_nn_basic_ops = get_all_nn_basic_operators()
    
    # Run benchmarks
    mx_nn_basic_op_results = run_op_benchmarks(mx_nn_basic_ops, dtype, ctx, profiler, int64_tensor, warmup, runs)
    return merge_map_list(rnn_relu_benchmark + rnn_tanh_benchmark + rnn_lstm_benchmark + rnn_gru_benchmark + [mx_nn_basic_op_results])
