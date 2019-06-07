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

"""Performance benchmark tests for MXNet NDArray Binary Operations - covers both broadcast and element_wise.
1. Operators are automatically fetched from MXNet operator registry.
2. Default Inputs are generated. See rules/default_params.py. You can override the default values.

Below 20 binary broadcast Operators are covered:

['broadcast_add', 'broadcast_div', 'broadcast_equal', 'broadcast_greater', 'broadcast_greater_equal',
'broadcast_hypot', 'broadcast_lesser', 'broadcast_lesser_equal', 'broadcast_logical_and',
'broadcast_logical_or', 'broadcast_logical_xor', 'broadcast_maximum', 'broadcast_minimum',
'broadcast_minus', 'broadcast_mod', 'broadcast_mul', 'broadcast_not_equal', 'broadcast_plus',
'broadcast_power', 'broadcast_sub']

Below 4 binary element_wise Operators are covered:
['elemwise_add', 'elemwise_mul', 'elemwise_sub', 'elemwise_div']

"""
import mxnet as mx

from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks
from benchmark.opperf.utils.op_registry_utils import get_all_broadcast_binary_operators, \
    get_all_elemen_wise_binary_operators


def run_mx_binary_broadcast_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the binary
    broadcast operators in MXNet.

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
    # Fetch all Binary Broadcast Operators
    mx_binary_broadcast_ops = get_all_broadcast_binary_operators()
    # Run benchmarks
    mx_binary_op_results = run_op_benchmarks(mx_binary_broadcast_ops, dtype, ctx, warmup, runs)
    return mx_binary_op_results


def run_mx_binary_element_wise_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    """Runs benchmarks with the given context and precision (dtype)for all the binary
    element_wise operators in MXNet.

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
    # Fetch all Binary Element_wise Operators
    mx_binary_element_wise_ops = get_all_elemen_wise_binary_operators()
    # Run benchmarks
    mx_binary_op_results = run_op_benchmarks(mx_binary_element_wise_ops, dtype, ctx, warmup, runs)
    return mx_binary_op_results
