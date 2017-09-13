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
import scipy.sparse as sp
import os
import time
import argparse

from mxnet.base import check_call, _LIB
from util import get_data, estimate_density

parser = argparse.ArgumentParser(description="Benchmark sparse operators",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-omp-threads', type=int, default=1, help='number of omp threads to set in MXNet')
args = parser.parse_args()

# some data information
kdda = {
    'data_mini': 'kdda.t.mini',
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
    'm': 200,
    'batch_size': [64]
}

avazu = {
    'data_mini': 'avazu-app.t.mini',
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
    'm': 500,
    'batch_size': [64, 128]
}


def measure_cost(repeat, f, *args, **kwargs):
    # start bench
    start = time.time()
    results = []
    for i in range(repeat):
        results.append(f(*args, **kwargs))
    for result in results:
        result.wait_to_read()
    end = time.time()
    diff = end - start
    return diff / repeat


def test_dot_real(data_dict):
    def get_iter(path, data_shape, batch_size):
        data_train = mx.io.LibSVMIter(data_libsvm=path,
                                      data_shape=data_shape,
                                      batch_size=batch_size)
        data_iter = iter(data_train)
        return data_iter

    data_dir = os.path.join(os.getcwd(), 'data')

    path = os.path.join(data_dir, data_dict['data_name'])
    if not os.path.exists(path):
        get_data(
            data_dir,
            data_dict['data_name'],
            data_dict['url'],
            data_dict['data_origin_name']
        )
        assert os.path.exists(path)

    k = data_dict['feature_dim']
    m = data_dict['m']
    density = estimate_density(path, data_dict['feature_dim'])

    mini_path = os.path.join(data_dir, data_dict['data_mini'])
    if not os.path.exists(mini_path):
        os.system("head -n 2000 %r > %r" % (path, mini_path))
        assert os.path.exists(mini_path)

    print "Running Benchmarking on %r data" % data_dict['data_mini']
    for batch_size in data_dict['batch_size']:  # iterator through different batch size of choice
        print "batch_size is %d" % batch_size
        # model
        data_shape = (k, )
        train_iter = get_iter(mini_path, data_shape, batch_size)
        weight = mx.nd.random.uniform(low=0, high=1, shape=(k, m))

        csr_data = []
        dns_data = []
        num_batch = 0
        for batch in train_iter:
            data = train_iter.getdata()
            csr_data.append(data)
            dns_data.append(data.tostype('default'))
            num_batch += 1
        bag_of_data = [csr_data, dns_data]
        num_repeat = 5
        costs = []
        for d in bag_of_data:
            weight.wait_to_read()
            cost = 0.
            count = 0
            for d_batch in d:
                d_batch.wait_to_read()
                cost += measure_cost(num_repeat, mx.nd.dot, d_batch, weight)
                count += 1
            costs.append(cost/count)
        t_sparse = costs[0]
        t_dense = costs[1]
        ratio = t_dense / t_sparse
        print('density(%)\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse')
        fmt = "%0.4f\t\t%d\t%d\t%d\t%0.2f\t\t\t%0.4f\t%0.6f"
        print(fmt % (density * 100, batch_size, m, k, ratio, t_dense, t_sparse))


def test_dot_synthetic():
    """benchmark mx.nd.dot(sparse_ndarray, dense_ndarray) with given density.
    `t_sparse` is the time cost of dot(csr, dns), while `t_dense` is the time cost
    of dot(dns, dns), with the same matrix except that it is in default storage type.
    """
    def measure_cost_forward_baseline(repeat, dot, lhs, rhs):
        start = time.time()
        for i in range(repeat):
            dot(lhs, rhs)
        end = time.time()
        diff = end - start
        return diff / repeat

    def measure_cost_backward_baseline(repeat, dot, transpose, lhs, rhs):
        start = time.time()
        for i in range(repeat):
            dot(transpose(lhs), rhs)
        end = time.time()
        diff = end - start
        return diff / repeat

    def bench_dot_forward(m, k, n, density, ctx, repeat):
        set_default_context(ctx)
        dns = mx.nd.random.uniform(shape=(k, n)).copyto(ctx)
        data_shape = (m, k)
        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.tostype('default')
        rhs_dns_np = dns.asnumpy()
        lhs_csr_sp = sp.csr_matrix(dns_data.asnumpy())  # csr in scipy
        lhs_dns_np = lhs_csr_sp.tostype('default')

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns)
            costs.append(cost)
        ratio = costs[0] / costs[1]

        costs_baseline = []
        cost = measure_cost_forward_baseline(repeat, np.dot, lhs_dns_np, rhs_dns_np)
        costs_baseline.append(cost)
        cost = measure_cost_forward_baseline(repeat, sp.spmatrix.dot, lhs_csr_sp, rhs_dns_np)
        costs_baseline.append(cost)
        ratio_baseline = costs_baseline[0] / costs_baseline[1]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.2f\t\t\t%0.2f\t%0.5f\t\t%0.2f\t\t\t\t%0.6f\t%0.5f"
        print(fmt % (density * 100, str(ctx), n, m, k, ratio, costs[0], costs[1],
                     ratio_baseline, costs_baseline[0], costs_baseline[1]))

    def bench_dot_backward(m, k, n, density, ctx, repeat):
        set_default_context(ctx)
        dns = mx.nd.random.uniform(shape=(m, n)).copyto(ctx)
        data_shape = (m, k)
        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.tostype('default')
        rhs_dns_np = dns.asnumpy()
        lhs_csr_sp = sp.csr_matrix(dns_data.asnumpy())
        lhs_dns_np = lhs_csr_sp.tostype('default')

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns, transpose_a=True)
            costs.append(cost)
        ratio = costs[0] / costs[1]

        costs_baseline = []
        cost = measure_cost_backward_baseline(repeat, np.dot, np.transpose, lhs_dns_np, rhs_dns_np)
        costs_baseline.append(cost)
        cost = measure_cost_backward_baseline(repeat, sp.spmatrix.dot, sp.spmatrix.transpose, lhs_csr_sp, rhs_dns_np)
        costs_baseline.append(cost)
        ratio_baseline = costs_baseline[0] / costs_baseline[1]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.2f\t\t\t%0.2f\t%0.5f\t\t%0.2f\t\t\t\t%0.6f\t%0.5f"
        print(fmt % (density * 100, str(ctx), n, m, k, ratio, costs[0], costs[1],
                     ratio_baseline, costs_baseline[0], costs_baseline[1]))

    print("A = sparse NDArray of shape(m, k)")
    print("B = dense NDArray of shape(k, n)")
    print("dot_forward\tdot(csr, dns)")
    print('density(%)\tcontext\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse'
          '\tt_scipy_dense/t_scipy_sparse\tt_scipy_dense\tt_scipy_sparse')

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    # TODO(haibin) make these runtime options
    m = 512
    k = [50000, 100000]
    n = [64, 128]
    density = [1.00, 0.90, 0.70, 0.50, 0.30, 0.20, 0.10, 0.07, 0.05, 0.02, 0.01, 0.005, 0.001]
    num_repeat = 10
    # contexts = [mx.cpu(), mx.gpu(0)]
    contexts = [mx.cpu()]
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_forward(m, k[i], n[i], den, ctx, num_repeat)

    print("dot_backward\tdot(csr.T, dns)")
    print('density(%)\tcontext\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse'
          '\tt_scipy_dense/t_scipy_sparse\tt_scipy_dense\tt_scipy_sparse')
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_backward(m, k[i], n[i], den, ctx, num_repeat)


if __name__ == "__main__":
    test_dot_real(avazu)
    test_dot_real(kdda)
    test_dot_synthetic()
