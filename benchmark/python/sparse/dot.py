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

import os
import time
import argparse
import subprocess
import scipy.sparse as sp

import mxnet as mx
import numpy as np
import numpy.random as rnd
from mxnet.test_utils import rand_ndarray, set_default_device, assert_almost_equal, get_bz2_data
from mxnet.base import check_call, _LIB
from util import estimate_density

PARSER = argparse.ArgumentParser(description="Benchmark sparse operators",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PARSER.add_argument('--num-omp-threads', type=int,
                    default=1, help='number of omp threads to set in MXNet')
PARSER.add_argument('--gpu', action='store_true',
                    help="to be run on gpu")
# TODO: Use logging later
PARSER.add_argument('--verbose', action='store_true',
                    help="Verbose output")
ARGS = PARSER.parse_args()

# some data information
KDDA = {
    'data_mini': 'kdda.t.mini',
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
    'm': [1, 8, 32],
    'batch_size': [64],
    'default_index': {'batch_size': 0,
                      'output_dim': 2},
    'num_batches': 10
}

AVAZU = {
    'data_mini': 'avazu-app.t.mini',
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
    'm': [1, 1000, 2000],
    'batch_size': [128, 256],
    'default_index': {'batch_size': 0,
                      'output_dim': 1},
    'num_batches': 10
}

CRITEO = {
    'data_mini': 'criteo.t.mini',
    'data_name': 'criteo.t',
    'data_origin_name': 'criteo.t.bz2',
    'url': "https://s3-us-west-2.amazonaws.com/sparse-dataset/criteo.t.bz2",
    'feature_dim': 8388621,
    'm': [1, 8, 16, 32, 64],
    'batch_size': [64, 128],
    'default_index': {'batch_size': 1,
                      'output_dim': 3},
    'num_batches': 10
}

SYNTHETIC1 = {
    'feature_dim': [1000000],
    'm': [256, 1000],
    'density': [0.001, 0.005, 0.01, 0.02, 0.05,
                0.1, 0.2, 0.5, 0.65],
    'batch_size': [64, 128],
    'default_index': {'batch_size': 1,
                      'density': 2,
                      'output_dim': 1,
                      'feature_dim': 0},
    'num_repeat': 10
}

SYNTHETIC2 = {
    'feature_dim': [8000000, 16000000],
    'm': [1, 32],
    'density': [0.001, 0.005, 0.01, 0.02, 0.05,
                0.1, 0.2, 0.5, 0.65],
    'batch_size': [64, 128],
    'default_index': {'batch_size': 1,
                      'density': 2,
                      'output_dim': 1,
                      'feature_dim': 0},
    'num_repeat': 10
}

def measure_cost(repeat, scipy_trans_lhs, scipy_dns_lhs, func_name, *args, **kwargs):
    """Measure time cost of running a function
    """
    mx.nd.waitall()
    args_list = []
    for arg in args:
        args_list.append(arg)
    start = time.time()
    if scipy_trans_lhs:
        args_list[0] = np.transpose(args_list[0]) if scipy_dns_lhs else sp.spmatrix.transpose(args_list[0])
    for _ in range(repeat):
        func_name(*args_list, **kwargs)
    mx.nd.waitall()
    end = time.time()
    diff = end - start
    return diff / repeat


def _get_iter(path, data_shape, batch_size):
    data_train = mx.io.LibSVMIter(data_libsvm=path,
                                  data_shape=data_shape,
                                  batch_size=batch_size)
    data_iter = iter(data_train)
    return data_iter


def _line_count(path):
    return int(subprocess.check_output('wc -l {}'.format(path), shell=True).split()[0])


def _compare_sparse_dense(data_dir, file_name, mini_file_name, feature_dim,
                          output_dim, density, batch_size, num_batches=3, num_repeat=5, transpose=False,
                          rsp=False):

    def create_mini_path(mini_path, path, num_batches):
        """Samples batches of size: batch_size, total number: num_batches
        from the dataset files for running benchmarks"""
        if not os.path.exists(mini_path):
            last = _line_count(path) - num_batches * batch_size
            last = last if last >= 1 else 1
            start = int(rnd.uniform(1, last))
            os.system("sed -n '{},{}p' {} > {}".format(
                start, start + num_batches * batch_size, repr(path), repr(mini_path)))
            assert os.path.exists(mini_path)

    def run_benchmark(mini_path):
        """Run benchmarks
        """
        data_shape = (feature_dim, )
        train_iter = _get_iter(mini_path, data_shape, batch_size)
        weight_row_dim = batch_size if transpose else feature_dim
        weight_shape = (weight_row_dim, output_dim)
        if not rsp:
            weight = mx.nd.random.uniform(low=0, high=1, shape=weight_shape)
        else:
            weight = rand_ndarray(weight_shape, "row_sparse", density=0.05, distribution="uniform")
        total_cost = {}
        average_cost = {}
        count = 0
        total_cost["sparse"] = 0.
        total_cost["dense"] = 0.
        for _ in train_iter:
            csr_data = train_iter.getdata()
            dns_data = csr_data.tostype('default')
            cost_sparse = measure_cost(num_repeat, False, False, mx.nd.sparse.dot, csr_data, weight, transpose_a=transpose)
            cost_dense = measure_cost(num_repeat, False, False, mx.nd.dot, dns_data, weight, transpose_a=transpose)
            total_cost["sparse"] += cost_sparse
            total_cost["dense"] += cost_dense
            count = count + 1
        average_cost["sparse"] = total_cost["sparse"] / count
        average_cost["dense"] = total_cost["dense"] / count
        return (average_cost["sparse"], average_cost["dense"])

    def print_result(average_cost_sparse, average_cost_dense):
        """Print result of comparison between sparse and dense
        """
        ratio = average_cost_dense / average_cost_sparse
        fmt = '{:15.4f} {:10d} {:10d} {:10d} {:20.2f} {:15.2f} {:15.2f} {:10} {:10}'
        print(fmt.format(density * 100, batch_size, output_dim, feature_dim,
                         ratio, average_cost_dense*1000, average_cost_sparse*1000,
                         transpose, rsp))

    mini_path = os.path.join(data_dir, mini_file_name)
    path = os.path.join(data_dir, file_name)
    create_mini_path(mini_path, path, num_batches)
    average_cost_sparse, average_cost_dense = run_benchmark(mini_path)
    print_result(average_cost_sparse, average_cost_dense)


def test_dot_real(data_dict):
    """Dot operator testing with real datasets"""
    data_dir = os.path.join(os.getcwd(), 'data')

    path = os.path.join(data_dir, data_dict['data_name'])
    if not os.path.exists(path):
        get_bz2_data(
            data_dir,
            data_dict['data_name'],
            data_dict['url'],
            data_dict['data_origin_name']
        )
        assert os.path.exists(path)

    k = data_dict['feature_dim']
    m = data_dict['m']
    batch_size_list = data_dict['batch_size']

    default_output_index = data_dict['default_index']['output_dim']
    default_batch_size_index = data_dict['default_index']['batch_size']
    density = estimate_density(path, data_dict['feature_dim'])
    num_batches = data_dict['num_batches']

    assert default_batch_size_index < len(batch_size_list)
    assert default_output_index < len(m)
    if ARGS.verbose:
        print(f"Running Benchmarking on {repr(data_dict['data_mini'])} data")
    print('{:>15} {:>10} {:>10} {:>10} {:>20} {:>15} {:>15} {:>10} {:>10}'.format('density(%)',
                                                                                  'n',
                                                                                  'm',
                                                                                  'k',
                                                                                  't_dense/t_sparse',
                                                                                  't_dense(ms)',
                                                                                  't_sparse(ms)',
                                                                                  'is_transpose',
                                                                                  'rhs_rsp'))

    for output_dim in m:
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, output_dim, density,
                              batch_size_list[default_batch_size_index], num_batches)
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, output_dim, density,
                              batch_size_list[default_batch_size_index], num_batches,
                              transpose=True)
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, output_dim, density,
                              batch_size_list[default_batch_size_index], num_batches, rsp=True)

    for batch_size in batch_size_list:
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, m[default_output_index], density, batch_size, num_batches)
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, m[default_output_index], density, batch_size, num_batches,
                              transpose=True)
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, output_dim, density,
                              batch_size_list[default_batch_size_index], num_batches, rsp=True)


def test_dot_synthetic(data_dict):
    """benchmark sparse mxnet dot and scipy dot operator with matrices of given density.
    `t_sparse` is the runtime of the invoked sparse dot operator in ms, while `t_dense` is the
    runtime of dot(dns, dns), with the same matrices except that they are in default storage type.
    """
    # Benchmark MXNet and Scipys dot operator
    def bench_dot(lhs_shape, rhs_shape, lhs_stype, rhs_stype,
                  lhs_den, rhs_den, trans_lhs, ctx, num_repeat=10, fw="mxnet", distribution="uniform"):
        set_default_device(ctx)
        assert fw == "mxnet" or fw == "scipy"
        # Set funcs
        dot_func_sparse = mx.nd.sparse.dot if fw == "mxnet" else sp.spmatrix.dot
        dot_func_dense = mx.nd.dot if fw == "mxnet" else np.dot
        # Create matrix instances
        lhs_nd = rand_ndarray(lhs_shape, lhs_stype, density=lhs_den, distribution=distribution)
        # only uniform distribution supported for rhs
        if rhs_stype == 'csr':
            rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den, distribution=distribution)
        else:
            rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den, distribution="uniform")
        lhs_dns = None
        rhs_dns = None
        dense_cost = None
        sparse_cost = None

        if fw == "mxnet":
            lhs_dns = lhs_nd if lhs_stype == 'default' else lhs_nd.tostype('default')
            rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.tostype('default')
            # One warm up run, verify correctness
            out = dot_func_sparse(lhs_nd, rhs_dns, trans_lhs)
            out_expected = dot_func_dense(lhs_dns, rhs_dns, trans_lhs)
            assert_almost_equal(out.asnumpy(), out_expected.asnumpy(), rtol=1e-1, atol=1e-1)
            sparse_cost = measure_cost(num_repeat, False, False, dot_func_sparse, lhs_nd, rhs_nd, trans_lhs)
            dense_cost = measure_cost(num_repeat, False, False, dot_func_dense, lhs_dns, rhs_dns, trans_lhs)
        else:
            lhs_dns = lhs_nd.asnumpy()
            rhs_dns = rhs_nd.asnumpy()
            lhs_nd = sp.csr_matrix(lhs_nd.asnumpy())
            rhs_nd = rhs_nd.asnumpy()
            # One warm up run, verify correctness
            lhs_nd_copy = sp.spmatrix.transpose(lhs_nd) if trans_lhs else lhs_nd
            out = dot_func_sparse(lhs_nd_copy, rhs_dns)
            sparse_cost = measure_cost(num_repeat, trans_lhs, False, dot_func_sparse, lhs_nd, rhs_nd)
            dense_cost = measure_cost(num_repeat, trans_lhs, True, dot_func_dense, lhs_dns, rhs_dns)

        speedup = dense_cost / sparse_cost
        # Print results
        m = lhs_shape[0]
        k = lhs_shape[1]
        n = rhs_shape[1]
        result_pattern = '{:15.1f} {:15.1f} {:>10} {:8d} {:8d} {:8d} {:13.2f} {:13.2f} {:8.2f}'
        results = result_pattern.format(lhs_den*100,
                                        rhs_den*100,
                                        str(ctx),
                                        m,
                                        k,
                                        n,
                                        sparse_cost*1000,
                                        dense_cost*1000,
                                        speedup)
        print(results)

    def print_benchmark_info(lhs, rhs, lhs_trans, fw):
        trans_str = "^T" if lhs_trans else ""
        print("========================================================")
        print(f"  {fw} sparse dot benchmark: dot({lhs}, {rhs}) = {rhs}  ")
        print(
            f"  (matrix multiplication: (m x k){trans_str} * (k x n) = m x n)  ")
        print("========================================================")
        headline_pattern = '{:>15} {:>15} {:>10} {:>8} {:>8} {:>8} {:>13} {:>13} {:>8}'
        headline = headline_pattern.format('lhs_density(%)',
                                           'rhs_density(%)',
                                           'context',
                                           'm', 'k', 'n',
                                           't_sparse(ms)',
                                           't_dense(ms)',
                                           'speedup')
        print(headline)

    def run_benchmark(ctx=None, lhs="csr", lhs_trans=False, rhs="dns", fw="mxnet", rhs_density=1,
                      distribution="uniform"):

        if rhs_density > 1 or rhs_density < 0:
            raise ValueError("rhs_density has to be between 0 and 1")

        print_benchmark_info(lhs, rhs, lhs_trans, fw)

        if rhs == "csr":
            lhs_stype = "default"
            rhs_stype = "csr"
            assert (lhs_stype == 'default'), "Only dot(default, csr) supported"
            # Arrange dimensions according to use case. For below csr will have num_rows << num_cols
            feature_dim_list = data_dict['batch_size']
            batch_size_list = data_dict['m']
            output_dim_list = data_dict['feature_dim']
            density_list = data_dict['density']
            default_output_index = data_dict['default_index']['feature_dim']
            default_density_index = data_dict['default_index']['density']
            default_feature_index = data_dict['default_index']['batch_size']
            default_batch_size_index = data_dict['default_index']['output_dim']
            num_repeat = data_dict['num_repeat']

        else:
            lhs_stype = "csr"
            rhs_stype = "row_sparse" if rhs == "rsp" else "default"

            feature_dim_list = data_dict['feature_dim']
            output_dim_list = data_dict['m']
            batch_size_list = data_dict['batch_size']
            density_list = data_dict['density']

            default_output_index = data_dict['default_index']['output_dim']
            default_batch_size_index = data_dict['default_index']['batch_size']
            default_feature_index = data_dict['default_index']['feature_dim']
            default_density_index = data_dict['default_index']['density']
            num_repeat = data_dict['num_repeat']

        for output_dim in output_dim_list:
            if lhs_trans:
                output_row_dim = batch_size_list[default_batch_size_index]
            else:
                output_row_dim = feature_dim_list[default_feature_index]
            bench_dot((batch_size_list[default_batch_size_index],
                       feature_dim_list[default_feature_index]),
                      (output_row_dim, output_dim),
                      lhs_stype, rhs_stype,
                      density_list[default_density_index], rhs_density,
                      lhs_trans, ctx, num_repeat=num_repeat,
                      fw=fw, distribution=distribution)

        for feature_dim in feature_dim_list:
            if lhs_trans:
                output_row_dim = batch_size_list[default_batch_size_index]
            else:
                output_row_dim = feature_dim
            bench_dot((batch_size_list[default_batch_size_index], feature_dim),
                      (output_row_dim, output_dim_list[default_output_index]),
                      lhs_stype, rhs_stype, density_list[default_density_index], rhs_density,
                      lhs_trans, ctx, num_repeat=num_repeat, fw=fw, distribution=distribution)

        for batch_size in batch_size_list:
            if lhs_trans:
                output_row_dim = batch_size
            else:
                output_row_dim = feature_dim_list[default_feature_index]
            bench_dot((batch_size, feature_dim_list[default_feature_index]),
                      (output_row_dim,
                       output_dim_list[default_output_index]),
                      lhs_stype, rhs_stype, density_list[default_density_index],
                      rhs_density, lhs_trans, ctx, num_repeat=num_repeat,
                      fw=fw, distribution=distribution)

        for density in density_list:
            if lhs_trans:
                output_row_dim = batch_size_list[default_batch_size_index]
            else:
                output_row_dim = feature_dim_list[default_feature_index]
            bench_dot((batch_size_list[default_batch_size_index],
                       feature_dim_list[default_feature_index]),
                      (output_row_dim,
                       output_dim_list[default_output_index]),
                      lhs_stype, rhs_stype, density, density, lhs_trans, ctx,
                      num_repeat=num_repeat, fw=fw, distribution=distribution)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(ARGS.num_omp_threads)))
    context = mx.gpu() if ARGS.gpu else mx.cpu()
    # TODO(anirudh): make the data dicts to config which can be passed at runtime
    distributions = ["uniform", "powerlaw"]
    for distribution in distributions:
        run_benchmark(context, lhs="csr",
                      rhs="default", lhs_trans=False,
                      fw="mxnet", rhs_density=1,
                      distribution=distribution)
        run_benchmark(context, lhs="csr",
                      rhs="default", lhs_trans=True,
                      fw="mxnet", rhs_density=1,
                      distribution=distribution)
        run_benchmark(context, lhs="csr",
                      rhs="rsp", lhs_trans=False,
                      fw="mxnet", rhs_density=0.05,
                      distribution=distribution)
        run_benchmark(context, lhs="default",
                      rhs="csr", lhs_trans=False,
                      fw="mxnet", rhs_density=0.001,
                      distribution=distribution)
        if not ARGS.gpu:
            run_benchmark(context, lhs="csr",
                          rhs="default", lhs_trans=False,
                          fw="scipy", rhs_density=1,
                          distribution=distribution)
            run_benchmark(context, lhs="csr",
                          rhs="default", lhs_trans=True,
                          fw="scipy", rhs_density=1,
                          distribution=distribution)


if __name__ == "__main__":
    begin_time = time.time()
    test_dot_real(KDDA)
    test_dot_real(AVAZU)
    test_dot_real(CRITEO)
    test_dot_synthetic(SYNTHETIC1)
    test_dot_synthetic(SYNTHETIC2)
    total_time = time.time() - begin_time
    print(f"total time is {total_time}")
