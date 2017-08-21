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

import ctypes

from mxnet.test_utils import *
import os
import time
import argparse

from mxnet.base import check_call, _LIB

parser = argparse.ArgumentParser(description="Benchmark cast storage operators",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-omp-threads', type=int, default=1, help='number of omp threads to set in MXNet')
args = parser.parse_args()

def measure_cost(repeat, f, *args, **kwargs):
    start = time.time()
    results = []
    for i in range(repeat):
        (f(*args, **kwargs)).wait_to_read()
    end = time.time()
    diff = end - start
    return diff / repeat


def run_cast_storage_synthetic():
    def dense_to_sparse(m, n, density, ctx, repeat, stype):
        set_default_context(ctx)
        data_shape = (m, n)
        dns_data = rand_ndarray(data_shape, stype, density).tostype('default')
        dns_data.wait_to_read()

        # do one warm up run, verify correctness
        assert same(mx.nd.cast_storage(dns_data, stype).asnumpy(), dns_data.asnumpy())

        # start benchmarking
        cost = measure_cost(repeat, mx.nd.cast_storage, dns_data, stype)
        results = '{:10.1f} {:>10} {:8d} {:8d} {:10.2f}'.format(density*100, str(ctx), m, n, cost*1000)
        print(results)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))

    # params
    # m           number of rows
    # n           number of columns
    # density     density of the matrix
    # num_repeat  number of benchmark runs to average over
    # contexts    mx.cpu(), mx.gpu()
    #             note: benchmark different contexts separately; to benchmark cpu, compile without CUDA
    # benchmarks  dns_to_csr, dns_to_rsp
    m = [  512,    512]
    n = [50000, 100000]
    density = [1.00, 0.80, 0.60, 0.40, 0.20, 0.10, 0.05, 0.02, 0.01]
    num_repeat = 10
    contexts = [mx.gpu()]
    benchmarks = ["dns_to_csr", "dns_to_rsp"]

    # run benchmark
    for b in benchmarks:
        stype = ''
        print("==================================================")
        if b is "dns_to_csr":
            stype = 'csr'
            print(" cast_storage benchmark: dense to csr, size m x n ")
        elif b is "dns_to_rsp":
            stype = 'row_sparse'
            print(" cast_storage benchmark: dense to rsp, size m x n ")
        else:
            print("invalid benchmark: %s" %b)
            continue
        print("==================================================")
        headline = '{:>10} {:>10} {:>8} {:>8} {:>10}'.format('density(%)', 'context', 'm', 'n', 'time(ms)')
        print(headline)
        for i in range(len(n)):
            for ctx in contexts:
                for den in density:
                    dense_to_sparse(m[i], n[i], den, ctx, num_repeat, stype)
            print("")
        print("")


if __name__ == "__main__":
    run_cast_storage_synthetic()
