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

import logging

import mxnet as mx
from mxnet import nd

from .ndarray_utils import get_mx_ndarray, nd_forward_and_profile, nd_forward_backward_and_profile
from .common_utils import merge_map_list
from .op_registry_utils import prepare_op_inputs
from benchmark.opperf.rules.default_params import PARAMS_OF_TYPE_NDARRAY


def _prepare_op_inputs(inputs, run_backward, dtype, ctx):
    kwargs_list = []

    for inp in inputs:
        kwargs = {}
        for key, value in inp.items():
            if key in PARAMS_OF_TYPE_NDARRAY:
                kwargs[key] = get_mx_ndarray(ctx=ctx, in_tensor=value,
                                             dtype=dtype,
                                             initializer=nd.normal,
                                             attach_grad=run_backward)
            else:
                kwargs[key] = value
        kwargs_list.append(kwargs)

    return kwargs_list


def _run_nd_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list):
    if run_backward:
        benchmark_helper_func = nd_forward_backward_and_profile
    else:
        benchmark_helper_func = nd_forward_and_profile

    # Warm up, ignore the profiler output
    _, _ = benchmark_helper_func(op, warmup, **kwargs_list[0])

    # Run Benchmarks
    op_benchmark_result = {op.__name__: []}
    logging.info(f"Begin Benchmark - {op.__name__}")
    for idx, kwargs in enumerate(kwargs_list):
        _, profiler_output = benchmark_helper_func(op, runs, **kwargs)

        # Add inputs used for profiling this operator into result
        profiler_output["inputs"] = inputs[idx]
        op_benchmark_result[op.__name__].append(profiler_output)
    logging.info(f"Complete Benchmark - {op.__name__}")
    return op_benchmark_result


def run_performance_test(ops, inputs, run_backward=True,
                         dtype='float32', ctx=mx.cpu(),
                         warmup=10, runs=50):
    """Run operator benchmark for given operator or list of operators, ops, with the given inputs.

    Returns benchmark results as a list of dictionary where each dictionary represents benchmarks result per operator.
    key -> name of the operator and value -> map of results (forward time, backward time, time spent in memory
    operations.

    Parameters
    ----------
    ops: [Str]
        One or list of operators to benchmark. Should be an NDArray operator.
    inputs: map
        Inputs for operator. Key should be name of parameter for operator.
        Example: inputs = {"lhs": (1024, 1024), "rhs": (1024, 1024)} for mx.nd.add
    run_backward: Boolean, Default is True
        Should we have backward operator benchmarks.
    dtype: Str, default 'float32'
        Precision to use for input tensors. Defaults to float32. Example: 'float32', 'int64'
    ctx: mx.ctx, default mx.cpu()
        Context to use for benchmarks. Default to mx.cpu()
    warmup: int, default 10
        Number of warmup runs
    runs: int, default 50
        Number of runs for capturing benchmark results

    Returns
    -------
    List of dictionary of benchmark results. key -> name of the operator, Value is benchmark results.

    """
    kwargs_list = _prepare_op_inputs(inputs, run_backward, dtype, ctx)

    if not isinstance(ops, list):
        ops = [ops]

    op_benchmark_result = []
    for op in ops:
        if hasattr(mx.nd, op.__name__):
            benchmark_result = _run_nd_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list)
        else:
            raise ValueError("Unknown NDArray operator provided to benchmark. -  ", op.__name__)
        op_benchmark_result.append(benchmark_result)
    return op_benchmark_result


def run_op_benchmarks(ops, dtype, ctx, warmup, runs):
    # For each operator, run benchmarks
    mx_op_benchmark_results = []
    for _, op_params in ops.items():
        # Prepare inputs for the operator
        inputs = prepare_op_inputs(op_params)
        # Run benchmarks
        cur_op_res = run_performance_test(op_params["nd_op_handle"],
                                          run_backward=op_params["has_backward"],
                                          dtype=dtype, ctx=ctx,
                                          inputs=inputs,
                                          warmup=warmup, runs=runs)
        mx_op_benchmark_results += cur_op_res

    # Prepare combined results for all operators
    mx_op_benchmark_results = merge_map_list(mx_op_benchmark_results)
    return mx_op_benchmark_results
