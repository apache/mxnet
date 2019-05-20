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

from .ndarray_utils import get_mx_ndarray, nd_forward_backward_and_profile
from .gluon_utils import block_forward_backward_and_profile


def _prepare_op_inputs(inputs, run_backward, dtype, ctx):
    kwargs_list = []

    for inp in inputs:
        kwargs = {}
        for key, value in inp.items():
            if key in ["lhs", "rhs", "data"]:
                kwargs[key] = get_mx_ndarray(ctx=ctx, in_tensor=value,
                                             dtype=dtype,
                                             initializer=nd.normal,
                                             attach_grad=run_backward)
            else:
                kwargs[key] = value
        kwargs_list.append(kwargs)

    return kwargs_list


def _run_nd_operator_performance_test(op, warmup, runs, inputs, kwargs_list):
    # Warm up, ignore the profiler output
    _, _ = nd_forward_backward_and_profile(op, warmup, **kwargs_list[0])

    # Run Benchmarks
    op_benchmark_result = {op.__name__: []}
    print("Begin Benchmark - ", op.__name__)
    for idx, kwargs in enumerate(kwargs_list):
        _, profiler_output = nd_forward_backward_and_profile(op, runs, **kwargs)

        # Add inputs used for profiling this operator into result
        profiler_output["inputs"] = inputs[idx]
        op_benchmark_result[op.__name__].append(profiler_output)
    print("Complete Benchmark - ", op.__name__)
    return op_benchmark_result


def _run_gluon_block_performance_test(op, ctx, warmup, runs, inputs, kwargs_list):
    # Run Benchmarks
    op_benchmark_result = {op.__name__: []}
    print("Begin Benchmark - ", op.__name__)
    for idx, kwargs in enumerate(kwargs_list):
        # Inputs will data and parameters required to create a block
        data = kwargs['data']
        del kwargs['data']
        # Create and initialize the block
        block = op(**kwargs)
        block.initialize(ctx=ctx)

        # Warm up, ignore profiler output
        _, _ = block_forward_backward_and_profile(block=block, runs=warmup, x=data)

        _, profiler_output = block_forward_backward_and_profile(block=block, runs=runs, x=data)

        # Add inputs used for profiling this operator into result
        profiler_output["inputs"] = inputs[idx]
        op_benchmark_result[op.__name__].append(profiler_output)
    print("Complete Benchmark - ", op.__name__)
    return op_benchmark_result


def run_performance_test(ops, inputs, run_backward=True,
                         dtype='float32', ctx=mx.cpu(),
                         warmup=10, runs=50):
    """Run operator benchmark for given operator or list of operators, ops, with the given inputs.

    Returns benchmark results as a list of dictionary where each dictionary represents benchmarks result per operator.
    key -> name of the operator and value -> map of results (forward time, backward time, time spent in memory
    operations.

    :param ops: One or list of operators to benchmark. Can be an NDArray operator or a Gluon Block
    :param inputs: map, Inputs for operator. Key should be name of parameter for operator.
                   Example: inputs = {"lhs": (1024, 1024), "rhs": (1024, 1024)} for mx.nd.add
    :param run_backward: Default is True. Should we have backward operator benchmarks.
    :param dtype: Precision to use for input tensors. Defaults to float32. Example: 'float32', 'int64'
    :param ctx: Context to use for benchmarks. Default to mx.cpu()
    :param warmup: Number of warmup runs
    :param runs: Number of runs for capturing benchmark results
    :return: List of dictionary of benchmark results. key -> name of the operator, Value is benchmark results.

    """
    kwargs_list = _prepare_op_inputs(inputs, run_backward, dtype, ctx)

    if not isinstance(ops, list):
        ops = [ops]

    op_benchmark_result = []
    for op in ops:
        if hasattr(mx.nd, op.__name__):
            benchmark_result = _run_nd_operator_performance_test(op, warmup, runs, inputs, kwargs_list)
        elif issubclass(op, mx.gluon.Block):
            benchmark_result = _run_gluon_block_performance_test(op, ctx, warmup, runs, inputs, kwargs_list)
        op_benchmark_result.append(benchmark_result)
    return op_benchmark_result
