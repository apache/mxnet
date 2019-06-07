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
from benchmark.opperf.utils.common_utils import merge_map_list

"""Performance benchmark tests for MXNet NDArray Activation Operators.

MXNet NDArray Pooling Operators

1. MaxPool1D
2. MaxPool2D
3. SumPool1D
4. SumPool2D
4. AvgPool1D
5. AvgPool2D
6. GlobalMaxPool1D
7. GlobalMaxPool2D
8. GlobalAvgPool1D
9. GlobalAvgPool2D
10.GlobalSumPool1D
11.GlobalSumPool2D

(Under the hood uses mx.nd.pooling)
"""


def run_pooling_operators_benchmarks(ctx=mx.cpu(), dtype='float32', warmup=10, runs=50):
    pool_types = ['avg', 'max', 'sum']
    global_pool_types = [0, 1]

    # Run 1D and 2D Pooling performance runs
    pool1d_benchmark_res = []
    pool2d_benchmark_res = []
    for pool_type in pool_types:
        for global_pool in global_pool_types:
            pool1d_benchmark_res += run_performance_test([nd.Pooling],
                                                         run_backward=True,
                                                         dtype=dtype,
                                                         ctx=ctx,
                                                         inputs=[{"data": (32, 3, 256),
                                                                  "kernel": 3,
                                                                  "pool_type": pool_type,
                                                                  "global_pool": global_pool,
                                                                  "stride": 1,
                                                                  "pad": 1,
                                                                  "layout": 'NCW'},
                                                                 {"data": (32, 3, 64),
                                                                  "kernel": 3,
                                                                  "pool_type": pool_type,
                                                                  "global_pool": global_pool,
                                                                  "stride": 1,
                                                                  "pad": 1,
                                                                  "layout": 'NCW'}
                                                                 ],
                                                         warmup=warmup,
                                                         runs=runs)
            pool2d_benchmark_res += run_performance_test([nd.Pooling],
                                                         run_backward=True,
                                                         dtype=dtype,
                                                         ctx=ctx,
                                                         inputs=[{"data": (32, 3, 256, 256),
                                                                  "kernel": (3, 3),
                                                                  "pool_type": pool_type,
                                                                  "global_pool": global_pool,
                                                                  "stride": (1, 1),
                                                                  "pad": (0, 0),
                                                                  "layout": 'NCHW'},
                                                                 {"data": (32, 3, 64, 64),
                                                                  "kernel": (3, 3),
                                                                  "pool_type": pool_type,
                                                                  "global_pool": global_pool,
                                                                  "stride": (1, 1),
                                                                  "pad": (0, 0),
                                                                  "layout": 'NCHW'}
                                                                 ],
                                                         warmup=warmup,
                                                         runs=runs)
    # Prepare combined results
    mx_pooling_op_results = merge_map_list(pool1d_benchmark_res + pool2d_benchmark_res)
    return mx_pooling_op_results
