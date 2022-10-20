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
    run_mx_binary_element_wise_operators_benchmarks, run_mx_binary_misc_operators_benchmarks
from benchmark.opperf.nd_operations.gemm_operators import run_gemm_operators_benchmarks
from benchmark.opperf.nd_operations.random_sampling_operators import run_mx_random_sampling_operators_benchmarks
from benchmark.opperf.nd_operations.reduction_operators import run_mx_reduction_operators_benchmarks
from benchmark.opperf.nd_operations.sorting_searching_operators import run_sorting_searching_operators_benchmarks
from benchmark.opperf.nd_operations.nn_activation_operators import run_activation_operators_benchmarks
from benchmark.opperf.nd_operations.nn_conv_operators import run_pooling_operators_benchmarks, \
    run_convolution_operators_benchmarks, run_transpose_convolution_operators_benchmarks
from benchmark.opperf.nd_operations.nn_basic_operators import run_nn_basic_operators_benchmarks
from benchmark.opperf.nd_operations.nn_optimizer_operators import run_optimizer_operators_benchmarks
from benchmark.opperf.nd_operations.indexing_routines import run_indexing_routines_benchmarks
from benchmark.opperf.nd_operations.nn_loss_operators import run_loss_operators_benchmarks
from benchmark.opperf.nd_operations.linalg_operators import run_linalg_operators_benchmarks
from benchmark.opperf.nd_operations.misc_operators import run_mx_misc_operators_benchmarks
from benchmark.opperf.nd_operations.array_manipulation_operators import run_rearrange_operators_benchmarks, \
    run_shape_operators_benchmarks, run_expanding_operators_benchmarks, run_rounding_operators_benchmarks, \
    run_join_split_operators_benchmarks

from benchmark.opperf.utils.common_utils import merge_map_list, save_to_file
from benchmark.opperf.utils.op_registry_utils import get_operators_with_no_benchmark, \
    get_current_runtime_features


def run_all_mxnet_operator_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', int64_tensor='off', warmup=25, runs=100):
    """Run all the MXNet operators (NDArray) benchmarks.

    Returns
    -------
    Dictionary of benchmark results.
    """
    mxnet_operator_benchmark_results = []

    # *************************MXNET TENSOR OPERATOR BENCHMARKS*****************************

    # Run all Unary operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_unary_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Binary Broadcast, element_wise, and miscellaneous operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_binary_broadcast_operators_benchmarks(ctx=ctx,
                                                                                         dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))
    mxnet_operator_benchmark_results.append(run_mx_binary_element_wise_operators_benchmarks(ctx=ctx,
                                                                                            dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    mxnet_operator_benchmark_results.append(run_mx_binary_misc_operators_benchmarks(ctx=ctx,
                                                                                         dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all GEMM operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_gemm_operators_benchmarks(ctx=ctx,
                                                                          dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Random sampling operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_random_sampling_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Reduction operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_reduction_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Sorting and Searching operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_sorting_searching_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Indexing routines benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_indexing_routines_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Array Rearrange operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_rearrange_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Array Shape Manipulation operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_shape_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Array Expansion operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_expanding_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Array Rounding operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_rounding_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Array Join & Split operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_join_split_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # ************************ MXNET NN OPERATOR BENCHMARKS ****************************

    # Run all basic NN operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_nn_basic_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Activation operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_activation_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Pooling operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_pooling_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Convolution operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_convolution_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Optimizer operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_optimizer_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Transpose Convolution operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_transpose_convolution_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all NN loss operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_loss_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Run all Miscellaneous operations benchmarks with default input values
    mxnet_operator_benchmark_results.append(run_mx_misc_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

    # Linear Algebra operators do not work with int64 tensor data. Issue tracked here: https://github.com/apache/incubator-mxnet/issues/17716
    if int64_tensor == 'off':
        # Run all Linear Algebra operations benchmarks with default input values
        mxnet_operator_benchmark_results.append(run_linalg_operators_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs))

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

    parser.add_argument('--int64-tensor', type=str, default='off',
                        help='Run performance tests with large tensor input'
                             'data (dimension >= 2**32) or standard input data.'
                             'Valid Inputs - on, off')

    parser.add_argument('-w', '--warmup', type=int, default=25,
                        help='Number of times to run for warmup.'
                             'Valid Inputs - positive integers')

    parser.add_argument('-r', '--runs', type=int, default=100,
                        help='Number of runs to capture benchmark results.'
                             'Valid Inputs - positive integers') 

    args = parser.parse_args()
    logging.info(f"Running MXNet operator benchmarks with the following options: {args}")
    assert not os.path.isfile(args.output_file),\
        f"Output file {args.output_file} already exists."

    # 2. RUN BENCHMARKS
    ctx = _parse_mxnet_context(args.ctx)
    dtype = args.dtype
    profiler = args.profiler
    int64_tensor = args.int64_tensor
    warmup = args.warmup
    runs = args.runs
    benchmark_results = run_all_mxnet_operator_benchmarks(ctx=ctx, dtype=dtype, profiler=profiler, int64_tensor=int64_tensor, warmup=warmup, runs=runs)

    # Sort benchmark results alphabetically by op name
    final_benchmark_results = dict()
    for key in sorted(benchmark_results.keys()):
        final_benchmark_results[key] = benchmark_results[key]

    # 3. PREPARE OUTPUTS
    run_time_features = get_current_runtime_features()
    save_to_file(final_benchmark_results, args.output_file, args.output_format, run_time_features, profiler)

    # 4. Generate list of MXNet operators not covered in benchmarks
    ops_not_covered = get_operators_with_no_benchmark(final_benchmark_results.keys())
    for idx, op in enumerate(ops_not_covered):
        print(f"{idx}. {op}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
