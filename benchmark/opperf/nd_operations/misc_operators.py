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

"""Performance benchmark tests for MXNet NDArray Miscellaneous Operations.

Below 16 Miscellaneous Operators are covered:

['reset_arrays', 'multi_all_finite', 'multi_sum_sq', 'add_n', 'UpSampling', 'Custom', 'squeeze',
'all_finite', 'clip', 'multi_lars', 'SequenceReverse', 'SequenceLast', 'SequenceMask', 'cast_storage',
'cumsum', 'fill_element_0index']

"""

import mxnet as mx

from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks
from benchmark.opperf.utils.op_registry_utils import get_remaining_miscellaneous_operators

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

from benchmark.opperf.custom_operations.custom_operations import CustomAddOneProp


def run_mx_misc_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the miscellaneous
    operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    int64_tensor: str, default 'off'
        Input tensor size to use for tests (if on, dimensions >= 2**32)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """

    standard_inputs_array_ops = [{"args": [(1024, 1024)],
                                  "num_arrays": 1},
                                 {"args": [(10000, 1)],
                                  "num_arrays": 1},
                                 {"args": [(10000, 10)],
                                  "num_arrays": 1}]
    int64_tensor_inputs_array_ops = [{"args": [(2**32, 1)],
                                      "num_arrays":1}]
    standard_inputs_add_n = [{"args": [(1024, 1024)]},
                             {"args": [(10000, 1)]},
                             {"args": [(10000, 10)]}]
    int64_tensor_inputs_add_n = [{"args": [(2**16, 2**16)]}]
    standard_inputs_upsampling = [{"args": (32, 3, 256, 256),
                                   "scale": 2,
                                   "sample_type": "nearest"},
                                  {"args": (32, 3, 10000, 1),
                                   "scale": 4,
                                   "sample_type": "nearest"}]
    int64_tensor_inputs_upsampling = [{"args": (2**32 + 1, 1, 1, 1),
                                       "scale": 2,
                                       "sample_type": "nearest"}]
    standard_inputs_custom = [{"args": [(1024, 1024)],
                               "op_type": "CustomAddOne"},
                              {"args": [(10000, 1)],
                               "op_type": "CustomAddOne"},
                              {"args": [(10000, 10)],
                               "op_type": "CustomAddOne"}]
    int64_tensor_inputs_custom = [{"args": [(2**32 + 1, 1)],
                                   "op_type": "CustomAddOne"}]

    if int64_tensor == 'on':
        inputs_array_ops = int64_tensor_inputs_array_ops
        inputs_add_n = int64_tensor_inputs_add_n
        inputs_upsampling = int64_tensor_inputs_upsampling
        inputs_custom = int64_tensor_inputs_custom
    else:
        inputs_array_ops = standard_inputs_array_ops
        inputs_add_n = standard_inputs_add_n
        inputs_upsampling = standard_inputs_upsampling
        inputs_custom = standard_inputs_custom

    # Individual tests for ops with positional args
    array_ops_benchmark = run_performance_test([getattr(MX_OP_MODULE, "reset_arrays"),
                                                getattr(MX_OP_MODULE, "multi_all_finite"),
                                                getattr(MX_OP_MODULE, "multi_sum_sq")],
                                               run_backward=False,
                                               dtype=dtype,
                                               ctx=ctx,
                                               profiler=profiler,
                                               inputs=inputs_array_ops,
                                               warmup=warmup,
                                               runs=runs)
    add_n_benchmark = run_performance_test([getattr(MX_OP_MODULE, "add_n")],
                                           run_backward=True,
                                           dtype=dtype,
                                           ctx=ctx,
                                           profiler=profiler,
                                           inputs=inputs_add_n,
                                           warmup=warmup,
                                           runs=runs)
    # There are currently issus with UpSampling with bilinear interpolation.
    # track issue here: https://github.com/apache/mxnet/issues/9138
    upsampling_benchmark = run_performance_test([getattr(MX_OP_MODULE, "UpSampling")],
                                                run_backward=True,
                                                dtype=dtype,
                                                ctx=ctx,
                                                profiler=profiler,
                                                inputs=inputs_upsampling,
                                                warmup=warmup,
                                                runs=runs)
    # Create and register CustomAddOne operator for use in Custom op testing
    c = CustomAddOneProp()
    c.create_operator(ctx, [(1024,1024)], [dtype])
    custom_benchmark = run_performance_test([getattr(MX_OP_MODULE, "Custom")],
                                            run_backward=True,
                                            dtype=dtype,
                                            ctx=ctx,
                                            profiler=profiler,
                                            inputs=inputs_custom,
                                            warmup=warmup,
                                            runs=runs)

    # Fetch remaining Miscellaneous Operators
    mx_misc_ops = get_remaining_miscellaneous_operators()
    # Run benchmarks
    mx_misc_op_results = run_op_benchmarks(mx_misc_ops, dtype, ctx, profiler, int64_tensor, warmup, runs)
    return merge_map_list(array_ops_benchmark + add_n_benchmark + upsampling_benchmark + custom_benchmark + [mx_misc_op_results])
