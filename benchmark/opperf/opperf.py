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

import argparse

import mxnet as mx
from benchmark.opperf.tensor_operations.arithmetic_operations import run_arithmetic_operators_benchmarks
from benchmark.opperf.nn_operations.convolution_operations import run_convolution_operators_benchmarks
from benchmark.opperf.utils.common_utils import merge_map_list, save_to_file


def run_all_mxnet_operator_benchmarks(ctx=mx.cpu(), dtype='float32'):
    """Run all the MXNet operators (NDArray and Gluon) benchmarks.

    :return: Dictionary of benchmark results.
    """
    mxnet_operator_benchmark_results = []

    # *************************MXNET TENSOR OPERATOR BENCHMARKS*****************************

    # Run all Arithmetic operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_arithmetic_operators_benchmarks(ctx=ctx,
                                                                                dtype=dtype))

    # ************************ MXNET GLUON NN LAYERS BENCHMARKS ****************************

    # Run all Gluon Convolution Layers operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_convolution_operators_benchmarks(ctx=ctx,
                                                                                 dtype=dtype))

    # ****************************** PREPARE FINAL RESULTS ********************************
    final_benchmark_result_map = merge_map_list(mxnet_operator_benchmark_results)
    return final_benchmark_result_map


def _parse_mxnet_context(ctx):
    if not ctx:
        raise ValueError("Context cannot be null or empty")

    if ctx.lower() in ['cpu', 'gpu']:
        return mx.context.Context(ctx)
    elif ctx.lower().startwith('gpu('):
        device_id = int(ctx[4:-1])
        return mx.gpu(device_id)


if __name__ == '__main__':
    # CLI Parser

    # 1. GET USER INPUTS
    parser = argparse.ArgumentParser(
        description='Run all the MXNet operators (NDArray and Gluon) benchmarks')

    parser.add_argument('--ctx', type=str, default='cpu',
                        help='Global context to run all benchmarks. By default, cpu on a '
                             'CPU machine, gpu(0) on a GPU machine. '
                             'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
    parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
                                                                     'float32. Valid Inputs - float32, float64.')
    parser.add_argument('--output-format', type=str, default='json',
                        help='Benchmark result output format. By default, json. '
                             'Valid Inputs - json, md, csv')

    parser.add_argument('--output-file', type=str, default='./mxnet_operator_benchmarks.json',
                        help='Name and path for the '
                             'output file.')

    # TODO - Input validation
    user_options = parser.parse_args()
    print("Running MXNet operator benchmarks with the following options: ")
    print(user_options)

    # 2. RUN BENCHMARKS
    ctx = _parse_mxnet_context(user_options.ctx)
    dtype = user_options.dtype
    final_benchmark_results = run_all_mxnet_operator_benchmarks(ctx=ctx, dtype=user_options.dtype)

    # 3. PREPARE OUTPUTS
    save_to_file(final_benchmark_results, user_options.output_file, user_options.output_format)
