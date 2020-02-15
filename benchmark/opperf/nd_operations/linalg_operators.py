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

"""Performance benchmark tests for MXNet NDArray Linear Algebra Operations.

Below 17 Linear Algebra Operators are covered:

['linalg_potri', 'linalg_gemm2', 'linalg_extractdiag', 'linalg_trsm', 'linalg_gelqf', 'linalg_gemm', 'linalg_sumlogdiag',
'linalg_potrf', 'linalg_makediag', 'linalg_syrk', 'linalg_maketrian', 'linalg_trmm', 'linalg_extracttrian',
'linalg_slogdet', 'linalg_det', 'linalg_inverse', 'moments']

"""

import mxnet as mx

from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks
from benchmark.opperf.utils.op_registry_utils import get_all_linalg_operators

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

def run_linalg_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the linear algebra
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Individual tests for ops with specific requirements on input data
    # linalg_potrf requires a positive definite matrix as input
    linalg_potrf_benchmark = run_performance_test(getattr(MX_OP_MODULE, "linalg_potrf"),
                                                  run_backward=False,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  profiler=profiler,
                                                  inputs=[{"A": [[1, 0],
                                                                 [0, 1]]},
                                                          {"A": [[2, -1, 0],
                                                                 [-1, 2, -1],
                                                                 [0, -1, 2]]}],
                                                  warmup=warmup,
                                                  runs=runs)

    # Fetch all Linear Algebra Operators
    mx_linalg_ops = get_all_linalg_operators()
    # Run benchmarks
    mx_linalg_op_results = run_op_benchmarks(mx_linalg_ops, dtype, ctx, profiler, warmup, runs)
    return merge_map_list(linalg_potrf_benchmark + [mx_linalg_op_results])
