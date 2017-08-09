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
    mx.nd.waitall()
    start = time.time()
    for i in range(repeat):
        f(*args, **kwargs)
    mx.nd.waitall()
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
        weight = mx.nd.random_uniform(low=0, high=1, shape=(k, m))

        csr_data = []
        dns_data = []
        num_batch = 0
        for batch in train_iter:
            data = train_iter.getdata()
            csr_data.append(data)
            dns_data.append(data.todense())
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
                cost += measure_cost(True, num_repeat, mx.nd.dot, d_batch, weight)
                count += 1
            costs.append(cost/count)
        t_sparse = costs[0]
        t_dense = costs[1]
        ratio = t_dense / t_sparse
        print('density(%)\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse')
        fmt = "%0.4f\t\t%d\t%d\t%d\t%0.2f\t\t\t%0.4f\t%0.6f"
        print(fmt % (density * 100, batch_size, m, k, ratio, t_dense, t_sparse))


def test_dot_synthetic():
    """benchmark sparse mxnet dot and scipy dot operator with matrices of given density.
    `t_sparse` is the runtime of the invoked sparse dot operator in ms, while `t_dense` is the 
    runtime of dot(dns, dns), with the same matrices except that they are in default storage type.
    """
    # Benchmark MXNet's sparse dot operator
    def bench_mx_dot(lhs_shape, rhs_shape, lhs_stype, rhs_stype, lhs_den, rhs_den, trans_lhs, ctx, repeat):
        set_default_context(ctx)
        # Create matrix instances
        lhs_nd = rand_ndarray(lhs_shape, lhs_stype, density=lhs_den)
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den)
        lhs_dns = lhs_nd if lhs_stype == 'default' else lhs_nd.todense()
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.todense()
        # One warm up run, verify correctness
        out = mx.nd.dot(lhs_nd, rhs_dns, trans_lhs)
        out_expected = mx.nd.dot(lhs_dns, rhs_dns, trans_lhs)
        assert_almost_equal(out.asnumpy(), out_expected.asnumpy(), rtol=1e-2, atol=1e-3)
        # Start benchmarking
        lhs_nd.wait_to_read()
        rhs_nd.wait_to_read()
        sparse_cost = measure_cost(repeat, mx.nd.dot, lhs_nd, rhs_nd, trans_lhs)
        dense_cost = measure_cost(repeat, mx.nd.dot, lhs_dns, rhs_dns, trans_lhs)
        speedup = dense_cost / sparse_cost
        # Print results
        m = lhs_shape[0]
        k = lhs_shape[1]
        n = rhs_shape[1]
        results = '{:15.1f} {:15.1f} {:>10} {:8d} {:8d} {:8d} {:13.2f} {:13.2f} {:8.2f}'.format(lhs_den*100, rhs_den*100, str(ctx), m, k, n, sparse_cost*1000, dense_cost*1000, speedup)
        print(results)

    # Benchmark Scipy's sparse dot operator
    def bench_sp_dot(lhs_shape, rhs_shape, lhs_stype, rhs_stype, lhs_den, rhs_den, trans_lhs, ctx, repeat):
        set_default_context(ctx)
        assert default_context().device_type is 'cpu'
        assert lhs_stype is 'csr'
        assert rhs_stype is 'default'
        # Create matrix instances
        lhs_nd = rand_ndarray(lhs_shape, lhs_stype, density=lhs_den)
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den)
        lhs_nd.wait_to_read()
        rhs_nd.wait_to_read()
        lhs_dns_np = np.transpose(lhs_nd.asnumpy()) if trans_lhs else lhs_nd.asnumpy()
        rhs_dns_np = rhs_nd.asnumpy()
        lhs_csr_sp = sp.spmatrix.transpose(sp.csr_matrix(lhs_nd.asnumpy())) if trans_lhs else sp.csr_matrix(lhs_nd.asnumpy())
        # One warm up run
        out = sp.spmatrix.dot(lhs_csr_sp, rhs_dns_np)
        # Start benchmarking
        sparse_cost = measure_cost(repeat, sp.spmatrix.dot, lhs_csr_sp, rhs_dns_np)
        dense_cost = measure_cost(repeat, np.dot, lhs_dns_np, rhs_dns_np)
        speedup = dense_cost / sparse_cost
        # Print results
        m = lhs_shape[0]
        k = lhs_shape[1]
        n = rhs_shape[1]
        results = '{:15.1f} {:15.1f} {:>10} {:8d} {:8d} {:8d} {:13.2f} {:13.2f} {:8.2f}'.format(lhs_den*100, rhs_den*100, str(ctx), m, k, n, sparse_cost*1000, dense_cost*1000, speedup)
        print(results)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    # TODO(haibin): make these runtime options
    # params
    # m, n, k        rows and columns of lhs and rhs matrix
    #                forward  pass:  m x k    * k x n = m x n
    #                backward pass: (m x k)^T * m x n = k x n
    # density_lhs    density of the left-hand side matrix
    # density_rhs    density of the right-hand side matrix, if applicable
    # num_repeat     number of benchmark runs to average over
    # context        mx.cpu(), mx.gpu()
    #                note: benchmark different contexts separately; to benchmark cpu, compile without CUDA
    # mx_benchmarks  csr_dns, csr.T_dns, csr_rsp
    # sp_benchmarks  csr_dns, csr.T_dns
    #                note: scipy benchmarks are only conducted if context is mx.cpu()
    m = 512
    k = [50000, 100000]
    n = [64, 128]
    density_lhs = [0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
    density_rhs = [0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
    num_repeat = 10
    context = mx.gpu()
    mx_benchmarks = ["csr_dns", "csr.T_dns", "csr_rsp"]
    sp_benchmarks = ["csr_dns", "csr.T_dns"]

    headline = '{:>15} {:>15} {:>10} {:>8} {:>8} {:>8} {:>13} {:>13} {:>8}'.format('lhs_density(%)', 'rhs_density(%)', 'context', 'm', 'k', 'n', 't_sparse(ms)', 't_dense(ms)', 'speedup')
    if "csr_dns" in mx_benchmarks:
        print("==================================================")
        print("  mxnet sparse dot benchmark: dot(csr, dns) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_mx_dot((m, k[i]), (k[i], n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print ""

    if "csr_dns" in sp_benchmarks and mx.cpu() == context:
        print("==================================================")
        print("  scipy sparse dot benchmark: dot(csr, dns) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_sp_dot((m, k[i]), (k[i], n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print ""

    if "csr.T_dns" in mx_benchmarks:
        print("==================================================")
        print(" mxnet sparse dot benchmark: dot(csr.T, dns) = rsp")
        print("(matrix multiplication: (m x k)^T * m x n = k x n)")
        print("==================================================")
        print(headline)
        transpose_lhs = True
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_mx_dot((m, k[i]), (m, n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print ""

    if "csr.T_dns" in sp_benchmarks and mx.cpu() == context:
        print("==================================================")
        print(" scipy sparse dot benchmark: dot(csr.T, dns) = dns")
        print("(matrix multiplication: (m x k)^T * m x n = k x n)")
        print("==================================================")
        print(headline)
        transpose_lhs = True
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_sp_dot((m, k[i]), (m, n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print ""

    if "csr_rsp" in mx_benchmarks:
        print("==================================================")
        print("  mxnet sparse dot benchmark: dot(csr, rsp) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
              for d_rhs in density_rhs:
                bench_mx_dot((m, k[i]), (k[i], n[i]), 'csr', 'row_sparse', d_lhs, d_rhs, transpose_lhs, context, num_repeat)
              print ""
            print ""


if __name__ == "__main__":
    test_dot_synthetic()
    test_dot_real(avazu)
    test_dot_real(kdda)
