"""Should be run with valgrind to get memory consumption
   for sparse format storage and dot operators"""
import ctypes
import sys
import argparse
import mxnet as mx
from mxnet.test_utils import rand_ndarray
from mxnet.base import check_call, _LIB


def parse_args():
    """ Function to parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs-row-dim",
                        required=True,
                        help="Provide batch_size")
    parser.add_argument("--lhs-col-dim",
                        required=True,
                        help="Provide feature_dim")
    parser.add_argument("--rhs-col-dim",
                        required=True,
                        help="Provide output_dim")
    parser.add_argument("--density",
                        required=True,
                        help="Density for lhs")
    parser.add_argument("--num-omp-threads", type=int,
                        default=1, help="number of omp threads to set in MXNet")
    parser.add_argument("--lhs-stype", default="csr",
                        choices=["csr", "default", "row_sparse"],
                        help="stype for lhs",
                        required=True)
    parser.add_argument("--rhs-stype", default="default",
                        choices=["default", "row_sparse"],
                        help="rhs stype",
                        required=True)
    parser.add_argument("--only-storage",
                        action="store_true",
                        help="only storage")
    parser.add_argument("--rhs-density",
                        help="rhs_density")
    return parser.parse_args()


def main():
    args = parse_args()
    lhs_row_dim = int(args.lhs_row_dim)
    lhs_col_dim = int(args.lhs_col_dim)
    rhs_col_dim = int(args.rhs_col_dim)
    density = float(args.density)
    lhs_stype = args.lhs_stype
    rhs_stype = args.rhs_stype
    if args.rhs_density:
        rhs_density = float(args.rhs_density)
    else:
        rhs_density = density
    dot_func = mx.nd.sparse.dot if lhs_stype == "csr" else mx.nd.dot
    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    bench_dot(lhs_row_dim, lhs_col_dim, rhs_col_dim, density,
              rhs_density, dot_func, False, lhs_stype, rhs_stype, args.only_storage)

def bench_dot(lhs_row_dim, lhs_col_dim, rhs_col_dim, density,
              rhs_density, dot_func, trans_lhs, lhs_stype,
              rhs_stype, only_storage, distribution="uniform"):
    """ Benchmarking both storage and dot
    """
    lhs_nd = rand_ndarray((lhs_row_dim, lhs_col_dim), lhs_stype, density, distribution=distribution)
    if not only_storage:
        rhs_nd = rand_ndarray((lhs_col_dim, rhs_col_dim), rhs_stype,
                              density=rhs_density, distribution=distribution)
        out = dot_func(lhs_nd, rhs_nd, trans_lhs)
    mx.nd.waitall()


if __name__ == '__main__':
    sys.exit(main())
