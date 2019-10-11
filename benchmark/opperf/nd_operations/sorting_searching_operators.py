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
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks
from benchmark.opperf.utils.op_registry_utils import get_all_sorting_searching_operators


""" Performance benchmark tests for MXNet NDArray Sorting and Searching Operations
1. sort
2. argsort
3. topk
4. argmax
5. argmin
"""


def run_sorting_searching_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype)for all the sorting and searching
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Fetch all Random Sampling Operators
    mx_sort_search_ops = get_all_sorting_searching_operators()
    # Run benchmarks
    mx_sort_search_op_results = run_op_benchmarks(mx_sort_search_ops, dtype, ctx, profiler, warmup, runs)
    return mx_sort_search_op_results
