#!/usr/bin/env python3
#
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
#
# -*- coding: utf-8 -*-

"""Commandline utility to run operator benchmarks"""

import argparse
import logging
import os
import sys

import mxnet as mx

from benchmark.opperf.nd_operations.unary_operators import run_mx_unary_operators_benchmarks
from benchmark.opperf.nd_operations.binary_operators import run_mx_binary_broadcast_operators_benchmarks, \
    run_mx_binary_element_wise_operators_benchmarks
from benchmark.opperf.nd_operations.gemm_operators import run_gemm_operators_benchmarks
from benchmark.opperf.nd_operations.random_sampling_operators import run_mx_random_sampling_operators_benchmarks
from benchmark.opperf.nd_operations.reduction_operators import run_mx_reduction_operators_benchmarks
from benchmark.opperf.nd_operations.sorting_searching_operators import run_sorting_searching_operators_benchmarks
from benchmark.opperf.nd_operations.nn_activation_operators import run_activation_operators_benchmarks
from benchmark.opperf.nd_operations.nn_conv_operators import run_pooling_operators_benchmarks, \
    run_convolution_operators_benchmarks, run_transpose_convolution_operators_benchmarks
from benchmark.opperf.nd_operations.nn_basic_operators import run_nn_basic_operators_benchmarks
from benchmark.opperf.nd_operations.nn_optimizer_operators import run_optimizer_operators_benchmarks
from benchmark.opperf.nd_operations.array_rearrange import run_rearrange_operators_benchmarks

from benchmark.opperf.utils.common_utils import merge_map_list, save_to_file
from benchmark.opperf.utils.op_registry_utils import get_operators_with_no_benchmark, \
    get_current_runtime_features


def run_all_mxnet_operator_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native'):
    """Run all the MXNet operators (NDArray) benchmarks.

    Returns
    -------
    Dictionary of benchmark results.
    """
    mxnet_operator_benchmark_results = []

    # *************************MXNET TENSOR OPERATOR BENCHMARKS*****************************

    # Run all Unary operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_unary_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Binary Broadcast, element_wise operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_binary_broadcast_operators_benchmarks(ctx=ctx,
                                                                                         dtype=dtype, profiler=profiler))
    mxnet_operator_benchmark_results.append(run_mx_binary_element_wise_operators_benchmarks(ctx=ctx,
                                                                                            dtype=dtype, profiler=profiler))

    # Run all GEMM operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_gemm_operators_benchmarks(ctx=ctx,
                                                                          dtype=dtype, profiler=profiler))

    # Run all Random sampling operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_random_sampling_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Reduction operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_reduction_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Sorting and Searching operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_sorting_searching_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Array Rearrange operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_rearrange_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # ************************ MXNET NN OPERATOR BENCHMARKS ****************************

    # Run all basic NN operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_nn_basic_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Activation operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_activation_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Pooling operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_pooling_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Convolution operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_convolution_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

    # Run all Optimizer operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_optimizer_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))
    # Run all Transpose Convolution operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_transpose_convolution_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler))

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


def main():
    # 1. GET USER INPUTS
    parser = argparse.ArgumentParser(description='Run all the MXNet operator benchmarks')

    parser.add_argument('--ctx', type=str, default='cpu',
                        help='Global context to run all benchmarks. By default, cpu on a '
                             'CPU machine, gpu(0) on a GPU machine. '
                             'Valid Inputs - cpu, gpu, gpu(0), gpu(1)...')
    parser.add_argument('--dtype', type=str, default='float32', help='DType (Precision) to run benchmarks. By default, '
                                                                     'float32. Valid Inputs - float32, float64, int32, '
                                                                     'int64')
    parser.add_argument('-f', '--output-format', type=str, default='json',
                        choices=['json', 'md'],
                        help='Benchmark result output format. By default, json. '
                             'Valid Inputs - json, md')

    parser.add_argument('-o', '--output-file', type=str, default='./mxnet_operator_benchmarks.json',
                        help='Name and path for the '
                             'output file.')

    parser.add_argument('-p', '--profiler', type=str, default='native',
                        help='Use built-in CPP profiler (native) or Python'
                             'time module.'
                             'Valid Inputs - native, python')

    args = parser.parse_args()
    logging.info("Running MXNet operator benchmarks with the following options: {args}".format(args=args))
    assert not os.path.isfile(args.output_file),\
        "Output file {output_file} already exists.".format(output_file=args.output_file)

    # 2. RUN BENCHMARKS
    ctx = _parse_mxnet_context(args.ctx)
    dtype = args.dtype
    profiler = args.profiler
    final_benchmark_results = run_all_mxnet_operator_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler)

    # 3. PREPARE OUTPUTS
    run_time_features = get_current_runtime_features()
    save_to_file(final_benchmark_results, args.output_file, args.output_format, run_time_features, profiler)

    # 4. Generate list of MXNet operators not covered in benchmarks
    ops_not_covered = get_operators_with_no_benchmark(final_benchmark_results.keys())
    for idx, op in enumerate(ops_not_covered):
        print("{idx}. {op}".format(idx=idx, op=op))

    return 0


if __name__ == '__main__':
    sys.exit(main())
