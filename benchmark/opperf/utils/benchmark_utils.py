import json
import mxnet as mx
from mxnet import nd

from .ndarray_utils import get_mx_ndarray, nd_forward_backward_and_profile


def prepare_op_inputs(inputs, run_backward, dtype, ctx):
    kwargs = {}
    for key, value in inputs.items():
        if key in ["lhs", "rhs"]:
            kwargs[key] = get_mx_ndarray(ctx=ctx, in_tensor=value,
                                         dtype=dtype,
                                         initializer=nd.normal,
                                         attach_grad=run_backward)
        else:
            kwargs[key] = value

    return kwargs


def run_performance_test(op, inputs, run_backward=True,
                         dtype='float32', ctx=mx.cpu(),
                         warmup=10, runs=50):
    kwargs = prepare_op_inputs(inputs, run_backward, dtype, ctx)
    # TODO - Check if this is a Gluon or an NDArray operator being profiled.
    # Warm up, ignore profiler output
    _, _ = nd_forward_backward_and_profile(op, warmup, **kwargs)

    # Run Benchmarks
    _, profiler_output = nd_forward_backward_and_profile(op, runs, **kwargs)

    # Add inputs used for profiling this operator into result
    profiler_output["inputs"] = inputs
    op_benchmark_result = {op.__name__: [profiler_output]}
    json_out = json.dumps(op_benchmark_result, indent=4)
    print(json_out)
