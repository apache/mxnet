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

"""Performance benchmark tests for MXNet NDArray GEMM Operators.

1. dot
2. batch_dot

TODO
3. As part of default tests, following needs to be added:
    3.1 Sparse dot. (csr, default) -> row_sparse
    3.2 Sparse dot. (csr, row_sparse) -> default
    3.3 With Transpose of lhs
    3.4 With Transpose of rhs
4. 1D array: inner product of vectors
"""


def run_gemm_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the GEMM
    operators (dot, batch_dot) in MXNet.

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
    # Benchmark tests for dot and batch_dot operators
    dot_benchmark_res = run_performance_test(
        [nd.dot], run_backward=True,
        dtype=dtype, ctx=ctx,
        inputs=[{"lhs": (1024, 1024),
                 "rhs": (1024, 1024)},
                {"lhs": (1000, 10),
                 "rhs": (1000, 10),
                 "transpose_b": True},
                {"lhs": (1000, 1),
                 "rhs": (100, 1000),
                 "transpose_a": True,
                 "transpose_b": True}],
        warmup=warmup, runs=runs)

    batch_dot_benchmark_res = run_performance_test(
        [nd.batch_dot], run_backward=True,
        dtype=dtype, ctx=ctx,
        inputs=[{"lhs": (32, 1024, 1024),
                 "rhs": (32, 1024, 1024)},
                {"lhs": (32, 1000, 10),
                 "rhs": (32, 1000, 10),
                 "transpose_b": True},
                {"lhs": (32, 1000, 1),
                 "rhs": (32, 100, 1000),
                 "transpose_a": True,
                 "transpose_b": True}],
        warmup=warmup, runs=runs)

    # Prepare combined results for GEMM operators
    mx_gemm_op_results = merge_map_list(dot_benchmark_res + batch_dot_benchmark_res)
    return mx_gemm_op_results
