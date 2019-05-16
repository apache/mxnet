""" Performance benchmark tests for MXNet NDArray GEMM Operations

TODO

1. dot
2. batch_dot

3. As part of default tests, following needs to be added:
    3.1 Sparse dot. (csr, default) -> row_sparse
    3.2 Sparse dot. (csr, row_sparse) -> default
    3.3 With Transpose of lhs
    3.4 With Transpose of rhs
4. 1D array: inner product of vectors
"""
