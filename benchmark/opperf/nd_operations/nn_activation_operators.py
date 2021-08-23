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

from benchmark.opperf.utils.op_registry_utils import get_all_nn_activation_operators
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks

"""Performance benchmark tests for MXNet NDArray Activation Operators.

1. LeakyReLU
    1.1 elu
    1.2 selu
    1.3 leaky
    1.4 gelu
2. hard_sigmoid
3. Softmax
4. SoftmaxActivation
5. softmax
6. log_softmax
7. softmin
8. Activation
    8.1 relu
    8.2 sigmoid
    8.3 log_sigmoid
    8.4 mish
    8.5 softrelu
    8.6 softsign
    8.7 tanh

"""


def run_activation_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and input data size (int64_tensor) for all the activation
    operators (relu, sigmoid, softmax) in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Module to use for tracking benchmark excecution time
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

    # Fetch all NN Activation Operators
    mx_activation_ops = get_all_nn_activation_operators()

    # Run benchmarks
    mx_activation_op_results = run_op_benchmarks(mx_activation_ops, dtype, ctx, profiler, int64_tensor, warmup, runs)
    return mx_activation_op_results
    