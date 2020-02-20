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

from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.benchmark_utils import run_op_benchmarks
from benchmark.opperf.utils.op_registry_utils import get_all_optimizer_operators
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE

"""Performance benchmark tests for MXNet Neural Network Optimizer Update Operators.

1. Stochastic Gradient Descent (SGD)
    1.1 mp_sgd_update
    1.2 sgd_mom_update
    1.3 signsgd_update
    1.4 mp_sgd_mom_update
    1.5 sgd_update
2. signum_update
3. rmspropalex_update
4. ftml_update
5. rmsprop_update
6. ftrl_update
7. adam_update
8. preloaded_multi_*
    8.1 preloaded_multi_sgd_mom_update
    8.2 preloaded_multi_sgd_update
    8.3 preloaded_multi_mp_sgd_update
    8.4 preloaded_multi_mp_sgd_mom_update
9. lamb_*
    9.1 lamb_update_phase1
    9.2 lamb_update_phase2
10. multi_*
    10.1 multi_sgd_update
    10.2 multi_sgd_mom_update
    10.3 multi_mp_sgd_update
    10.4 multi_mp_sgd_mom_update
"""


def run_optimizer_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype) for all the neural network
    optimizer update operators in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Run independent tests for ops that need specific input data
    multi_mp_sgd_mom_res = run_performance_test([getattr(MX_OP_MODULE, "multi_mp_sgd_mom_update")],
                                                inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                                "args1": nd.random_normal(shape=(5,5)), "args2": nd.random_normal(shape=(5,5)),
                                                "args3": nd.random_normal(shape=(5,5)), "lrs": 0.1, "wds": 0.2,
                                                "out": nd.random_normal(shape=(5,5))}],run_backward=False)

    multi_sgd_mom_res = run_performance_test([getattr(MX_OP_MODULE, "multi_sgd_mom_update")],
                                             inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                             "args1": nd.random_normal(shape=(5,5)),"args2": nd.random_normal(shape=(5,5)),
                                             "lrs": 0.1, "wds": 0.2, "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    multi_sgd_res = run_performance_test([getattr(MX_OP_MODULE, "multi_sgd_update")],
                                         inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                         "args1": nd.random_normal(shape=(5,5)), "lrs": 0.1, "wds": 0.2,
                                         "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    multi_mp_sgd_res = run_performance_test([getattr(MX_OP_MODULE, "multi_mp_sgd_update")],
                                            inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                            "args1": nd.random_normal(shape=(5,5)),"args2": nd.random_normal(shape=(5,5)),
                                            "lrs": 0.1, "wds": 0.2, "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    preloaded_multi_mp_sgd_res = run_performance_test(
                                 [getattr(MX_OP_MODULE, "preloaded_multi_mp_sgd_update")],
                                 inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                          "args1": nd.random_normal(shape=(5,5)), "args2": nd.random_normal(shape=(5,5)),
                                          "args3": nd.random_normal(shape=(1)), "args4": nd.random_normal(shape=(1)),
                                          "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    preloaded_multi_sgd_mom_res = run_performance_test(
                                  [getattr(MX_OP_MODULE, "preloaded_multi_sgd_mom_update")],
                                  inputs=[{"args0": nd.random_normal(shape=(5,5)),
                                           "args1": nd.random_normal(shape=(5,5)), "args2": nd.random_normal(shape=(5,5)),
                                           "args3": nd.random_normal(shape=(1)), "args4": nd.random_normal(shape=(1)),
                                           "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    preloaded_multi_sgd_res = run_performance_test(
                              [getattr(MX_OP_MODULE, "preloaded_multi_sgd_update")],
                              inputs=[{"args0": nd.random_normal(shape=(5,5)), "args1": nd.random_normal(shape=(5,5)),
                                       "args4": nd.random_normal(shape=(1)), "args5": nd.random_normal(shape=(1)),
                                       "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    preloaded_multi_mp_sgd_mom_res = run_performance_test(
                                     [getattr(MX_OP_MODULE, "preloaded_multi_mp_sgd_mom_update")],
                                     inputs=[{"args0": nd.random_normal(shape=(5,5)), "args1": nd.random_normal(shape=(5,5)),
                                              "args2": nd.random_normal(shape=(5,5)), "args3": nd.random_normal(shape=(5,5)),
                                              "args4": nd.random_normal(shape=(1)), "args5": nd.random_normal(shape=(1)),
                                              "out": nd.random_normal(shape=(5,5))}], run_backward=False)

    # Fetch remaining optimizer operators
    mx_optimizer_ops = get_all_optimizer_operators()

    # Run benchmarks
    mx_optimizer_op_results = run_op_benchmarks(mx_optimizer_ops, dtype, ctx, profiler, warmup, runs)
    return merge_map_list(multi_sgd_mom_res + multi_sgd_mom_res + multi_sgd_res + multi_mp_sgd_res + preloaded_multi_mp_sgd_res +\
                          preloaded_multi_sgd_mom_res + preloaded_multi_mp_sgd_res + preloaded_multi_mp_sgd_mom_res +\
                          [mx_optimizer_op_results])
