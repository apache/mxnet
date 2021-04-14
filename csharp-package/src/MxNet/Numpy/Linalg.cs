using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public class Linalg
    {
        public ndarray matrix_rank(ndarray M, ndarray tol= null, bool hermitian= false)
        {
            return nd_np_ops.linalg.matrix_rank(M, tol, hermitian);
        }

        public (ndarray, ndarray, ndarray, ndarray) lstsq(ndarray a, ndarray b, string rcond= "warn")
        {
            return nd_np_ops.linalg.lstsq(a, b, rcond);
        }

        public ndarray pinv(ndarray a, float rcond= 1e-15f,bool hermitian= false)
        {
            return nd_np_ops.linalg.pinv(a, rcond, hermitian);
        }

        public ndarray norm(ndarray x, string ord= null, Shape axis= null, bool keepdims= false)
        {
            return nd_np_ops.linalg.norm(x, ord, axis, keepdims);
        }

        public ndarray svd(ndarray a)
        {
            return nd_np_ops.linalg.svd(a);
        }

        public ndarray cholesky(ndarray a)
        {
            return nd_np_ops.linalg.cholesky(a);
        }

        public ndarray qr(ndarray a, string mode = "reduced")
        {
            return nd_np_ops.linalg.qr(a, mode);
        }

        public ndarray inv(ndarray a)
        {
            return nd_np_ops.linalg.inv(a);
        }

        public ndarray det(ndarray a)
        {
            return nd_np_ops.linalg.det(a);
        }

        public ndarray slogdet(ndarray a)
        {
            return nd_np_ops.linalg.slogdet(a);
        }

        public ndarray solve(ndarray a, ndarray b)
        {
            return nd_np_ops.linalg.solve(a, b);
        }

        public ndarray tensorinv(ndarray a, int ind = 2)
        {
            return nd_np_ops.linalg.tensorinv(a, ind);
        }

        public ndarray tensorsolve(ndarray a, ndarray b, params int[] axes)
        {
            return nd_np_ops.linalg.tensorsolve(a, b, axes);
        }

        public ndarray eigvals(ndarray a)
        {
            return nd_np_ops.linalg.eigvals(a);
        }

        public ndarray eigvalsh(ndarray a, string UPLO = "L")
        {
            return nd_np_ops.linalg.eigvalsh(a, UPLO);
        }

        public (ndarray, ndarray) eig(ndarray a)
        {
            return nd_np_ops.linalg.eig(a);
        }

        public (ndarray, ndarray) eigh(ndarray a, string UPLO = "L")
        {
            return nd_np_ops.linalg.eigh(a, UPLO);
        }
    }
}
