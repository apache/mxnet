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

"""Performance benchmark tests for MXNet NDArray Unary Operations.
1. Operators are automatically fetched from MXNet operator registry.
2. Default Inputs are generated. See rules/default_params.py. You can override the default values.

Below 54 unary Operators are covered:

['BlockGrad', 'Flatten', 'abs', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh',
'argmax_channel', 'cbrt', 'ceil', 'cos', 'cosh', 'degrees', 'erf', 'erfinv', 'exp', 'expm1', 'fix', 'flatten',
'floor', 'gamma', 'gammaln', 'identity', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'make_loss', 'negative',
'ones_like', 'radians', 'rcbrt', 'reciprocal', 'relu', 'rint', 'round', 'rsqrt', 'shuffle', 'sigmoid', 'sign',
'sin', 'sinh', 'size_array', 'softsign', 'sqrt', 'square', 'stop_gradient', 'tan', 'tanh', 'trunc', 'zeros_like']

"""

import mxnet as mx

from benchmark.opperf.utils.op_registry_utils import get_all_unary_operators
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

def run_mx_unary_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context, precision (dtype), and input data size (int64_tensor) for all the unary
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

    standard_inputs = [{"args": [(1024, 1024)],
                        "num_outputs":1},
                       {"args": [(10000, 1)],
                        "num_outputs":1}]
    int64_tensor_inputs = [{"args": [(2**32, 1)],
                            "num_outputs":1}]

    if int64_tensor == 'on':
        inputs = int64_tensor_inputs
    else:
        inputs = standard_inputs

    # Run amp_multicast as it needs data as positional argument
    amp_multicast_benchmark = run_performance_test([getattr(MX_OP_MODULE, "amp_multicast")],
                                                   run_backward=True,
                                                   dtype=dtype,
                                                   ctx=ctx,
                                                   profiler=profiler,
                                                   inputs=inputs,
                                                   warmup=warmup,
                                                   runs=runs)

    # Fetch all Unary Operators
    mx_unary_broadcast_ops = get_all_unary_operators()

    # Run benchmarks
    mx_unary_op_results = run_op_benchmarks(mx_unary_broadcast_ops, dtype, ctx, profiler, int64_tensor, warmup, runs)
    return merge_map_list(amp_multicast_benchmark + [mx_unary_op_results])
