using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.ND.Numpy
{
    internal class Linalg
    {
        private static dynamic _api_internal = new _api_internals();

        public ndarray matrix_rank(ndarray M, ndarray tol= null, bool hermitian= false)
        {
            if (hermitian)
            {
                throw new NotImplementedException("hermitian is not supported yet...");
            }
            return _api_internal.matrix_rank(M: M, tol: tol, hermitian: hermitian, finfo_eps_32: float.Epsilon, finfo_eps_64: double.Epsilon);
        }

        public (ndarray, ndarray, ndarray, ndarray) lstsq(ndarray a, ndarray b, string rcond= "warn")
        {
            var list = (NDArrayList)_api_internal.lstsq(a: a, b: b, rcond: rcond, finfo_eps_32: float.Epsilon, finfo_eps_64: double.Epsilon, multi: true);
            var x = list[0];
            var residuals = list[1];
            var rank = list[2];
            var s = list[3];
            return (x, residuals, rank, s);
        }

        public ndarray pinv(ndarray a, float rcond= 1e-15f,bool hermitian= false)
        {
            if (hermitian)
            {
                throw new NotImplementedException("hermitian is not supported yet...");
            }

            return _api_internal.pinv(a: a, rcond: rcond, hermitian: hermitian);
        }

        public ndarray norm(ndarray x, string ord= null, Shape axis= null, bool keepdims= false)
        {
            int col_axis;
            int row_axis;
            if (axis == null && ord == null)
            {
                return _api_internal.norm(x, 2, null, keepdims, -2);
            }

            if (axis != null)
            {
                if (axis.Dimension == 2)
                {
                    if (new List<string> {
                            "inf",
                            "-inf"
                        }.Contains(ord))
                    {
                        row_axis = axis[0];
                        col_axis = axis[1];
                        if (!keepdims)
                        {
                            if (row_axis > col_axis)
                            {
                                row_axis -= 1;
                            }
                        }
                        if (ord == "inf")
                        {
                            return nd_np_ops.sum(nd_np_ops.abs(x), axis: col_axis, keepdims: keepdims).max(axis: row_axis, keepdims: keepdims);
                        }
                        else
                        {
                            return nd_np_ops.sum(nd_np_ops.abs(x), axis: col_axis, keepdims: keepdims).min(axis: row_axis, keepdims: keepdims);
                        }
                    }
                    if (new List<string> {
                            "1",
                            "-1"
                        }.Contains(ord))
                    {
                        row_axis = axis[0];
                        col_axis = axis[1];
                        if (!keepdims)
                        {
                            if (row_axis < col_axis)
                            {
                                col_axis -= 1;
                            }
                        }
                        if (ord == "1")
                        {
                            return nd_np_ops.sum(nd_np_ops.abs(x), axis: row_axis, keepdims: keepdims).max(axis: col_axis, keepdims: keepdims);
                        }
                        else if (ord == "-1")
                        {
                            return nd_np_ops.sum(nd_np_ops.abs(x), axis: row_axis, keepdims: keepdims).min(axis: col_axis, keepdims: keepdims);
                        }
                    }
                    if (new List<string> {
                            "2",
                            "-2"
                        }.Contains(ord))
                    {
                        return _api_internal.norm(x, ord, axis, keepdims, 0);
                    }
                    if (ord == null)
                    {
                        return _api_internal.norm(x, 2, axis, keepdims, 1);
                    }
                }

                throw new Exception("'axis' must be None, an integer or a tuple of integers.");
            }

            if (ord == "inf")
            {
                return nd_np_ops.max(nd_np_ops.abs(x), axis: axis[0], keepdims: keepdims);
            }
            else if (ord == "-inf")
            {
                return nd_np_ops.min(nd_np_ops.abs(x), axis: axis[0], keepdims: keepdims);
            }
            else if (ord == null)
            {
                return _api_internal.norm(x, 2, axis, keepdims, 1);
            }
            else if (ord == "2")
            {
                return _api_internal.norm(x, 2, axis, keepdims, -1);
            }
            else if (ord == "nuc")
            {
                return _api_internal.norm(x, 2, axis, keepdims, 2);
            }
            else if (new List<string> {
                    "fro",
                    "f"
                }.Contains(ord))
            {
                return _api_internal.norm(x, 2, axis, keepdims, 1);
            }
            else
            {
                return _api_internal.norm(x, ord, axis, keepdims, -1);
            }
        }

        public NDArrayList svd(ndarray a)
        {
            NDArrayList list = _api_internal.svd(a: a, multi: true);
            return list;
        }

        public ndarray cholesky(ndarray a)
        {
            return _api_internal.cholesky(a, true);
        }

        public NDArrayList qr(ndarray a, string mode = "reduced")
        {
            if (mode != null && mode != "reduced")
            {
                throw new NotImplementedException("Only default mode='reduced' is implemented.");
            }

            return (NDArrayList)_api_internal.qr(a: a, multi: true);
        }

        public ndarray inv(ndarray a)
        {
            return _api_internal.inv(a);
        }

        public ndarray det(ndarray a)
        {
            return _api_internal.det(a);
        }

        public NDArrayList slogdet(ndarray a)
        {
            return (NDArrayList)_api_internal.slogdet(a: a, multi: true);
        }

        public ndarray solve(ndarray a, ndarray b)
        {
            return _api_internal.solve(a, b);
        }

        public ndarray tensorinv(ndarray a, int ind = 2)
        {
            return _api_internal.tensorinv(a, ind);
        }

        public ndarray tensorsolve(ndarray a, ndarray b, params int[] axes)
        {
            return _api_internal.tensorsolve(a, b, axes);
        }

        public ndarray eigvals(ndarray a)
        {
            return _api_internal.eigvals(a);
        }

        public ndarray eigvalsh(ndarray a, string UPLO = "L")
        {
            return _api_internal.eigvalsh(a, UPLO);
        }

        public (ndarray, ndarray) eig(ndarray a)
        {
            var list = (NDArrayList)_api_internal.eig(a,  multi: true);
            var w = list[0];
            var v = list[1];
            return (w, v);
        }

        public (ndarray, ndarray) eigh(ndarray a, string UPLO = "L")
        {
            var list = (NDArrayList)_api_internal.eigh(a, multi: true);
            var w = list[0];
            var v = list[1];
            return (w, v);
        }
    }
}
