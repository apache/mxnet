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
from benchmark.opperf.rules.default_params import DEFAULT_DATA
from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

""" Performance benchmark tests for MXNet NDArray Sorting and Searching Operations
1. sort
2. argsort
3. topk
4. argmax
5. argmin
"""


def run_sorting_searching_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=25, runs=100):
    for data in DEFAULT_DATA:
        # Sort
        sort_benchmark_res = run_performance_test([getattr(MX_OP_MODULE, "sort")],
                                                  run_backward=False,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  inputs=[{"data": data}],
                                                  warmup=warmup,
                                                  runs=runs)
        # ArgSort
        argsort_benchmark_res = run_performance_test([getattr(MX_OP_MODULE, "argsort")],
                                                     run_backward=False,
                                                     dtype=dtype,
                                                     ctx=ctx,
                                                     inputs=[{"data": data}],
                                                     warmup=warmup,
                                                     runs=runs)

        # ArgMax
        argmax_benchmark_res = run_performance_test([getattr(MX_OP_MODULE, "argmax")],
                                                    run_backward=False,
                                                    dtype=dtype,
                                                    ctx=ctx,
                                                    inputs=[{"data": data, "axis": 0}],
                                                    warmup=warmup,
                                                    runs=runs)
        # ArgMin
        argmin_benchmark_res = run_performance_test([getattr(MX_OP_MODULE, "argmin")],
                                                    run_backward=False,
                                                    dtype=dtype,
                                                    ctx=ctx,
                                                    inputs=[{"data": data, "axis": 0}],
                                                    warmup=warmup,
                                                    runs=runs)

        # TopK
        topk_benchmark_res = run_performance_test([getattr(MX_OP_MODULE, "topk")],
                                                  run_backward=False,
                                                  dtype=dtype,
                                                  ctx=ctx,
                                                  inputs=[{"data": data, "k": 5, "axis": 0}],
                                                  warmup=warmup,
                                                  runs=runs)
    # Prepare combined results
    mx_sort_search_results = merge_map_list(
        sort_benchmark_res + argsort_benchmark_res + argmax_benchmark_res + argmin_benchmark_res + topk_benchmark_res)
    return mx_sort_search_results
