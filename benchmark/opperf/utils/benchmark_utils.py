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
import inspect

import mxnet as mx
from mxnet import nd
from mxnet import np

from .ndarray_utils import get_mx_ndarray, nd_forward_and_profile, nd_forward_backward_and_profile
from .common_utils import merge_map_list
from .op_registry_utils import prepare_op_inputs
from benchmark.opperf.rules.default_params import PARAMS_OF_TYPE_NDARRAY, PARAMS_OF_TYPE_NP_ARRAY
from .profiler_utils import cpp_profile, python_profile

no_backward = {'gather_nd', 'softmax_cross_entropy', 'linalg_gelqf', 'linalg_slogdet', 'moments', 'SequenceLast', 'Embedding'}

def _prepare_op_inputs(inputs, run_backward, dtype, ctx, module):
    mx.random.seed(41)
    kwargs_list = []
    if module == 'mxnet.numpy_extension' or module == 'mxnet.numpy':
        PARAMS_TYPE = PARAMS_OF_TYPE_NP_ARRAY
        get_array_fn = get_mx_np_ndarray
    else:
        PARAMS_TYPE = PARAMS_OF_TYPE_NDARRAY
        get_array_fn = get_mx_ndarray

    for inp in inputs:
        kwargs = {}
        for key, value in inp.items():
            if key in PARAMS_TYPE:
                kwargs[key] = get_array_fn(ctx=ctx, in_tensor=value,
                                           dtype=dtype,
                                           initializer=nd.normal,
                                           attach_grad=run_backward)
            else:
                kwargs[key] = value
        kwargs_list.append(kwargs)
    return kwargs_list

def get_mx_np_ndarray(ctx, in_tensor, dtype, initializer, attach_grad=True):
    """Helper function to prepare a MXNet Numpy NDArray tensor in given Context (ctx) of type (dtype).
    You can get a new Tensor by providing only "Shape" or "Numpy NDArray" or another MXNet NDArray as
    "in_tensor".

    NOTE: This is a sync call and waits for the Tensor to be created.

    Parameters
    ----------
    ctx: mx.ctx, default mx.cpu()
        Context of the new MXNet NDArray Tensor.
    in_tensor: Numpy NDArray or MXNet NDArray or Tuple of shape
        Can be a tuple of shape or Numpy NDArray or MXNet NDArray.
    dtype: str
        Precision or Dtype of the expected Tensor. Ex: "float32", "Int64"
    initializer:
        Function reference to the initialize to use. Ex: mx.nd.random.normal, mx.nd.zeros
    attach_grad: Boolean, default True
        To attach a gradient for the Tensor. Default is True.

    Returns
    -------
    MXNet NDArray Tensor.
    """
    if isinstance(in_tensor, int) or isinstance(in_tensor, float):
        return in_tensor

    if isinstance(in_tensor, tuple):
        nd_ndarray = get_mx_ndarray(ctx=ctx, in_tensor=in_tensor,
                                             dtype="float32",
                                             initializer=initializer,
                                             attach_grad=attach_grad)
        tensor = nd_ndarray.as_np_ndarray().astype(dtype=dtype)
    elif isinstance(in_tensor, list):
        tensor = np.array(in_tensor, ctx=ctx)
    elif isinstance(in_tensor, nd.NDArray):
        tensor = in_tensor.as_np_ndarray()
    elif isinstance(in_tensor, np.ndarray):
        tensor = in_tensor.as_in_context(ctx)
    else:
        raise ValueError("Invalid input type for creating input tensor. Input can be tuple() of shape or Numpy Array or"
                         " MXNet NDArray. Given - ", in_tensor)
    if attach_grad:
        tensor.attach_grad()

    tensor.wait_to_read()
    return tensor

def adjust_op_name(module, name):
    np_to_nd_func = {
        "batch_norm":           "BatchNorm",
        "fully_connected":      "FullyConnected",
        "activation":           "Activation",
        "convolution":          "Convolution" }
    nd_to_np_func = {
        "BatchNorm":            "batch_norm",
        "FullyConnected":       "fully_connected",
        "Activation":           "activation",
        "Convolution":          "convolution" }

    if (module == mx.nd and (hasattr(mx.np, name) or hasattr(mx.npx, name)) and name in np_to_nd_func.keys()):
        return np_to_nd_func[name]
    elif ((module == mx.np or module == mx.npx) and hasattr(mx.nd, name) and name in nd_to_np_func.keys()):
        return nd_to_np_func[name]
    else:
        return name

def parse_input_ndarray(input_dict):
    """Parse input for ndarray and extract array shape for better readability

    Parameters
    ----------
    input_dict : dict
         Dictionary of input

    Input Dictionary

    'inputs': {'weight':
    [[ 2.2122064   0.7740038   1.0434405   1.1839255   1.8917114 ]
     [-1.2347414  -1.771029   -0.45138445  0.57938355 -1.856082  ]
     [-1.9768796  -0.20801921  0.2444218  -0.03716067 -0.48774993]
     [-0.02261727  0.57461417  1.4661262   0.6862904   0.35496104]
     [ 1.0731696   0.12017461 -0.9711102  -0.77569664 -0.7882176 ]]
    <NDArray 5x5 @cpu(0)>, 'grad':
    [[ 0.7417728  -1.4734439  -1.0730928  -1.0424827  -1.3278849 ]
     [-1.4749662  -0.52414197  1.2662556   0.8950642  -0.6015945 ]
     [ 1.2040559  -0.9712193  -0.58256227  0.3717077   0.9300072 ]
     [-1.4225755  -0.5176199   2.0088325   0.2863085   0.5604595 ]
     [ 0.96975976 -0.52853745 -1.88909     0.65479124 -0.45481315]]
    <NDArray 5x5 @cpu(0)>, 'mean':
    [[ 0.32510808 -1.3002341   0.3679345   1.4534262   0.24154152]
     [ 0.47898006  0.96885103 -1.0218245  -0.06812762 -0.31868345]
     [-0.17634277  0.35655284  0.74419165  0.7787424   0.6087823 ]
     [ 1.0741756   0.06642842  0.8486986  -0.8003802  -0.16882208]
     [ 0.93632793  0.357444    0.77932847 -1.0103073  -0.39157307]]
    <NDArray 5x5 @cpu(0)>, 'var':
    [[ 1.3166187  -0.43292624  0.71535987  0.9254156  -0.90495086]
     [-0.074684    0.82254    -1.8785107   0.8858836   1.9118724 ]
     [ 0.33342266  0.11883813 -1.9198899  -0.67558455  1.007749  ]
     [-0.35391203  1.6323917  -0.33354783 -1.7378405   0.7737382 ]
     [ 0.89126545  3.2904532  -1.1976235   1.8938874  -0.5669272 ]]
    <NDArray 5x5 @cpu(0)>, 't': 1, 'wd': 0.1}

    Output
    {'inputs': {'weight': '<NDArray 5x5 @cpu(0)>', 'grad': '<NDArray 5x5 @cpu(0)>', 'mean': '<NDArray 5x5 @cpu(0)>', 'var': '<NDArray 5x5 @cpu(0)>', 't': 1, 'wd': 0.1}
    """
    no_new_line_input_dict=dict()
    for key,value in input_dict.items():
        if isinstance(value,nd.NDArray):
            # if value in input is NDArray then extract last line only
            val = str(value).split('\n')[-1]
            no_new_line_input_dict[key]=val
        else:
            no_new_line_input_dict[key]=value
    return no_new_line_input_dict


def _run_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list, profiler):
    if profiler == 'native':
        if run_backward:
            benchmark_helper_func = cpp_profile(nd_forward_backward_and_profile)
        else:
            benchmark_helper_func = cpp_profile(nd_forward_and_profile)
    elif profiler == 'python':
        if run_backward:
            benchmark_helper_func = python_profile(nd_forward_backward_and_profile)
        else:
            benchmark_helper_func = python_profile(nd_forward_and_profile)
    else:
        raise ValueError("Incorrect input for profiler. Valid input - 'python' or 'native'")

    # Warm up, ignore the profiler output
    _, _ = benchmark_helper_func(op, warmup, **kwargs_list[0])

    # Run Benchmarks
    op_benchmark_result = {op.__name__: []}
    logging.info(f"Begin Benchmark - {op.__name__}")

    for idx, kwargs in enumerate(kwargs_list):
        _, profiler_output = benchmark_helper_func(op, runs, **kwargs)

        # Add inputs used for profiling this operator into result
        # parse input if it contains ndarray, replace with shape info for better markdown readability
        new_inp = parse_input_ndarray(inputs[idx])
        profiler_output = merge_map_list([{"inputs": new_inp}] + [profiler_output])
        op_benchmark_result[op.__name__].append(profiler_output)
    logging.info(f"Complete Benchmark - {op.__name__}")
    return op_benchmark_result


def run_performance_test(ops, inputs, run_backward=True,
                         dtype='float32', ctx=mx.cpu(), profiler='native',
                         warmup=10, runs=50):
    """Run operator benchmark for given operator or list of operators, ops, with the given inputs.

    Returns benchmark results as a list of dictionary where each dictionary represents benchmarks result per operator.
    key -> name of the operator and value -> map of results (forward time, backward time, time spent in memory
    operations.

    Parameters
    ----------
    ops: [Str]
        One or list of operators to benchmark. Should be an NDArray, Numpy or Numpy_extension operator.
    inputs: map
        Inputs for operator. Key should be name of parameter for operator.
        Example: inputs = {"lhs": (1024, 1024), "rhs": (1024, 1024)} for mx.nd.add or
                 inputs = {"x1": (1024, 1024), "x2": (1024, 1024)} for mx.np.add
    run_backward: Boolean, Default is True
        Should we have backward operator benchmarks.
    dtype: Str, default 'float32'
        Precision to use for input tensors. Defaults to float32. Example: 'float32', 'int64'
    ctx: mx.ctx, default mx.cpu()
        Context to use for benchmarks. Default to mx.cpu()
    profiler: Str, default 'native'
        Type of profiler to run benchmarks. Default to 'native'
        Option - ['python', 'native']
    warmup: int, default 10
        Number of warmup runs
    runs: int, default 50
        Number of runs for capturing benchmark results

    Returns
    -------
    List of dictionary of benchmark results. key -> name of the operator, Value is benchmark results.

    Note: when run_performance_test is called on the nd.Embedding operator with run_backward=True, an error will
    be thrown. Track issue here: https://github.com/apache/mxnet/issues/11314
    """
    if not isinstance(ops, list):
        ops = [ops]

    op_benchmark_result = []
    for op in ops:
        if hasattr(mx.nd, op.__name__) or hasattr(mx.np, op.__name__) or hasattr(mx.npx, op.__name__):
            kwargs_list = _prepare_op_inputs(inputs, run_backward, dtype, ctx, op.__module__)
            benchmark_result = _run_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list, profiler)
        else:
            raise ValueError(f"Unknown {op.__module__} operator provided to benchmark. - {op.__name__}")
        op_benchmark_result.append(benchmark_result)
    return op_benchmark_result

def run_benchmark_operator(name, size = (128,128), additional_inputs = {},
                           dtype = 'float32', run_backward = False, ctx = mx.cpu(),
                           warmup=10, runs=50, profiler="native"):
    arg_list = {mx.nd: PARAMS_OF_TYPE_NDARRAY, mx.np: PARAMS_OF_TYPE_NP_ARRAY, mx.npx: PARAMS_OF_TYPE_NP_ARRAY}
    modules = [mx.nd, mx.np, mx.npx]
    responses = []
    for module in modules:
        name = adjust_op_name(module, name)
        if hasattr(module, name):
            function = getattr(module, name)
            args = inspect.signature(function).parameters.keys()
            inputs = {}
            for arg in args:
                if arg in additional_inputs.keys():
                    inputs.update({arg: additional_inputs[arg]})
                elif arg in arg_list[module]:
                    inputs.update({arg:size})
            res = run_performance_test(function, run_backward=run_backward, dtype=dtype, ctx=ctx,
                                       inputs=[inputs], warmup=warmup, runs=runs, profiler=profiler)
            responses.append(res)
        else:
            responses.append(str(module.__name__) + " does not have operator " + name)
    for i in range(len(modules)):
        print(modules[i].__name__)
        print(responses[i])

def run_op_benchmarks(ops, dtype, ctx, profiler, int64_tensor, warmup, runs):
    # Running im2col either forwards or backwards on GPU results in errors
    # track issue here: https://github.com/apache/mxnet/issues/17493
    gpu_disabled_ops = ['im2col']

    # For each operator, run benchmarks
    mx_op_benchmark_results = []
    for op, op_params in ops.items():
        if ctx == mx.cpu() or op not in gpu_disabled_ops:
            # Prepare inputs for the operator
            inputs = prepare_op_inputs(op, op_params, int64_tensor)

            # setting backward false for ops with known issue
            if op in no_backward:
                op_params["has_backward"] = False

            # Run benchmarks
            cur_op_res = run_performance_test(op_params["nd_op_handle"],
                                              run_backward=op_params["has_backward"],
                                              dtype=dtype, ctx=ctx,
                                              profiler=profiler,
                                              inputs=inputs,
                                              warmup=warmup, runs=runs)
            mx_op_benchmark_results += cur_op_res

    # Prepare combined results for all operators
    mx_op_benchmark_results = merge_map_list(mx_op_benchmark_results)
    return mx_op_benchmark_results
