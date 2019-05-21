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

"""Performance benchmark tests for MXNet NDArray Logical Operations

1. logical_and
2. logical_or
3. logical_xor
4. logical_not
"""


def run_logical_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the logical
    operators in MXNet.

    :param ctx: Context to run benchmarks
    :param dtype: Precision to use for benchmarks
    :param warmup: Number of times to run for warmup
    :param runs: Number of runs to capture benchmark results
    :return: Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Benchmark tests for logical_and, or, xor, not operators
    # Note: For backward pass, these nodes just create ZeroGrads, hence no use running backward pass.
    benchmark_res = run_performance_test([nd.logical_and, nd.logical_or, nd.logical_xor], run_backward=False,
                                         dtype=dtype, ctx=ctx,
                                         inputs=[{"lhs": (1024, 1024),
                                                  "rhs": (1024, 1024)},
                                                 {"lhs": (10000, 10),
                                                  "rhs": (10000, 10)},
                                                 {"lhs": (10000, 1),
                                                  "rhs": (10000, 100)}],
                                         warmup=warmup, runs=runs)

    benchmark_logical_not_res = run_performance_test(nd.logical_not, run_backward=False,
                                                     dtype=dtype, ctx=ctx,
                                                     inputs=[{"data": (1024, 1024)},
                                                             {"data": (10000, 10)},
                                                             {"data": (10000, 1)}],
                                                     warmup=warmup, runs=runs)

    # Prepare combined results for Comparison operators
    mx_logical_op_results = merge_map_list(benchmark_res + benchmark_logical_not_res)
    return mx_logical_op_results
