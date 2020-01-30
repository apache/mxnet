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
from benchmark.opperf.utils.op_registry_utils import get_all_rearrange_operators, \
    get_all_shape_operators, get_all_expanding_operators

"""Performance benchmark tests for MXNet Array Manipulation Operators.

Array Rearrange Operators
1. transpose
2. swapaxes (alias SwapAxis)
3. flip (alias reverse)
4. depth_to_space
5. space_to_depth

Array Shape Manipulation Operators
1. split (alias SliceChannel)
2. diag
3. reshape
4. reshape_like
5. size_array
6. shape_array

Array Expanding Operators
1. broadcast_axes (alias broadcast_axis)
2. broadcast_to
3. broadcast_like
4. repeat
5. tile
6. pad
7. expand_dims

"""


def run_rearrange_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the
    rearrange operators in MXNet.

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
    # Fetch all array rearrange operators
    mx_rearrange_ops = get_all_rearrange_operators()

    # Run benchmarks
    mx_rearrange_op_results = run_op_benchmarks(mx_rearrange_ops, dtype, ctx, profiler, warmup, runs)
    return mx_rearrange_op_results


def run_shape_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the
    array shape operators  in MXNet.

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
    # Fetch all array shape operators
    mx_shape_ops = get_all_shape_operators()

    # Run benchmarks
    mx_shape_op_results = run_op_benchmarks(mx_shape_ops, dtype, ctx, profiler, warmup, runs)
    return mx_shape_op_results


def run_expanding_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the
    array expanding operators  in MXNet.

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
    # Fetch all array expanding operators
    mx_expanding_ops = get_all_expanding_operators()

    # Run benchmarks
    mx_expanding_op_results = run_op_benchmarks(mx_expanding_ops, dtype, ctx, profiler, warmup, runs)
    return mx_expanding_op_results
