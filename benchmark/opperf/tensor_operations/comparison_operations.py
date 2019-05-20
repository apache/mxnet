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

"""Performance benchmark tests for MXNet NDArray Comparison Operations

1. lesser
2. lesser_equal
3. greater
4. greater_equal
5. equal
6. not_equal
"""


def run_comparison_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the comparison
    operators in MXNet.

    :param ctx: Context to run benchmarks
    :param dtype: Precision to use for benchmarks
    :param warmup: Number of times to run for warmup
    :param runs: Number of runs to capture benchmark results
    :return: Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Benchmark tests for Add, Sub, Mul, divide, modulo operators
    # Note: For backward pass, these nodes just create ZeroGrads, hence no use running backward pass.
    benchmark_res = run_performance_test(
        [nd.lesser, nd.lesser_equal, nd.greater, nd.greater_equal, nd.equal, nd.not_equal], run_backward=False,
        dtype=dtype, ctx=ctx,
        inputs=[{"lhs": (1024, 1024),
                 "rhs": (1024, 1024)},
                {"lhs": (10000, 10),
                 "rhs": (10000, 10)},
                {"lhs": (10000, 1),
                 "rhs": (10000, 100)}],
        warmup=warmup, runs=runs)

    # Prepare combined results for Comparison operators
    mx_comparison_op_results = merge_map_list(benchmark_res)
    return mx_comparison_op_results
