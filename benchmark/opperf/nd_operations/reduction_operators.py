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

"""Performance benchmark tests for MXNet NDArray Reduction Operations.
1. Operators are automatically fetched from MXNet operator registry.
2. Default Inputs are generated. See rules/default_params.py. You can override the default values.

Below 10 reduction Operators are covered:

['max', 'max_axis', 'mean', 'min', 'min_axis', 'nanprod', 'nansum', 'prod', 'sum', 'sum_axis']

"""

import mxnet as mx

from benchmark.opperf.utils.op_registry_utils import get_all_reduction_operators
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks


def run_mx_reduction_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the reduction
    operators in MXNet.

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
    # Fetch all Reduction Operators
    mx_reduction_broadcast_ops = get_all_reduction_operators()
    # Run benchmarks
    mx_reduction_op_results = run_op_benchmarks(mx_reduction_broadcast_ops, dtype, ctx, warmup, runs)
    return mx_reduction_op_results
