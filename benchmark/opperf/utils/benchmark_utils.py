import mxnet as mx
from mxnet import nd

from .ndarray_utils import get_mx_ndarray, nd_forward_backward_and_profile


def _prepare_op_inputs(inputs, run_backward, dtype, ctx):
    kwargs_list = []

    for inp in inputs:
        kwargs = {}
        for key, value in inp.items():
            if key in ["lhs", "rhs"]:
                kwargs[key] = get_mx_ndarray(ctx=ctx, in_tensor=value,
                                             dtype=dtype,
                                             initializer=nd.normal,
                                             attach_grad=run_backward)
            else:
                kwargs[key] = value
        kwargs_list.append(kwargs)

    return kwargs_list


def run_performance_test(op, inputs, run_backward=True,
                         dtype='float32', ctx=mx.cpu(),
                         warmup=10, runs=50):
    """Run operator benchmark for given operator, op, with given inputs.

    Returns benchmark results as a dictionary, where, key -> name of the operator,
    and value -> map of results (forward time, backward time, time spent in memory
    operations.

    :param op: Operator to benchmark. Can be an NDArray operator or a Gluon Block
    :param inputs: map, Inputs for operator. Key should be name of parameter for operator.
                   Example: inputs = {"lhs": (1024, 1024), "rhs": (1024, 1024)} for mx.nd.add
    :param run_backward: Default is True. Should we have backward operator benchmarks.
    :param dtype: Precision to use for input tensors. Defaults to float32. Example: 'float32', 'int64'
    :param ctx: Context to use for benchmarks. Default to mx.cpu()
    :param warmup: Number of warmup runs
    :param runs: Number of runs for capturing benchmark results
    :return: Dictionary of benchmark results. key -> name of the operator, Value is benchmark results.

    """
    kwargs_list = _prepare_op_inputs(inputs, run_backward, dtype, ctx)
    # TODO - Check if this is a Gluon or an NDArray operator being profiled.
    # Warm up, ignore profiler output
    _, _ = nd_forward_backward_and_profile(op, warmup, **kwargs_list[0])

    # Run Benchmarks
    op_benchmark_result = {op.__name__: []}
    for idx, kwargs in enumerate(kwargs_list):
        _, profiler_output = nd_forward_backward_and_profile(op, runs, **kwargs)

        # Add inputs used for profiling this operator into result
        profiler_output["inputs"] = inputs[idx]
        op_benchmark_result[op.__name__].append(profiler_output)

    return op_benchmark_result
