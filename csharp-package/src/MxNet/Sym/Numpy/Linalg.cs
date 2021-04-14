using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Sym.Numpy
{
    internal class Linalg
    {
        private static dynamic _api_internal = new _api_internals();

        public _Symbol matrix_rank(_Symbol M, _Symbol tol = null, bool hermitian = false)
        {
            if (hermitian)
            {
                throw new NotImplementedException("hermitian is not supported yet...");
            }
            return _api_internal.matrix_rank(M: M, tol: tol, hermitian: hermitian, finfo_eps_32: float.Epsilon, finfo_eps_64: double.Epsilon);
        }

        public (_Symbol, _Symbol, _Symbol, _Symbol) lstsq(_Symbol a, _Symbol b, string rcond = "warn")
        {
            var list = (SymbolList)_api_internal.lstsq(a: a, b: b, rcond: rcond, finfo_eps_32: float.Epsilon, finfo_eps_64: double.Epsilon, multi: true);
            var x = list[0];
            var residuals = list[1];
            var rank = list[2];
            var s = list[3];
            return (x, residuals, rank, s);
        }

        public _Symbol pinv(_Symbol a, float rcond = 1e-15f, bool hermitian = false)
        {
            if (hermitian)
            {
                throw new NotImplementedException("hermitian is not supported yet...");
            }

            return _api_internal.pinv(a: a, rcond: rcond, hermitian: hermitian);
        }

        public _Symbol norm(_Symbol x, string ord = null, Shape axis = null, bool keepdims = false)
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
                            return sym_np_ops.max(sym_np_ops.sum(sym_np_ops.abs(x), axis: col_axis, keepdims: keepdims), axis: row_axis, keepdims: keepdims);
                        }
                        else
                        {
                            return sym_np_ops.max(sym_np_ops.sum(sym_np_ops.abs(x), axis: col_axis, keepdims: keepdims), axis: row_axis, keepdims: keepdims);
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
                            return sym_np_ops.max(sym_np_ops.sum(sym_np_ops.abs(x), axis: row_axis, keepdims: keepdims), axis: col_axis, keepdims: keepdims);
                        }
                        else if (ord == "-1")
                        {
                            return sym_np_ops.min(sym_np_ops.sum(sym_np_ops.abs(x), axis: row_axis, keepdims: keepdims), axis: col_axis, keepdims: keepdims);
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
                return sym_np_ops.max(sym_np_ops.abs(x), axis: axis[0], keepdims: keepdims);
            }
            else if (ord == "-inf")
            {
                return sym_np_ops.min(sym_np_ops.abs(x), axis: axis[0], keepdims: keepdims);
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

        public SymbolList svd(_Symbol a)
        {
            SymbolList list = _api_internal.svd(a: a, multi: true);
            return list;
        }

        public _Symbol cholesky(_Symbol a)
        {
            return _api_internal.cholesky(a, true);
        }

        public SymbolList qr(_Symbol a, string mode = "reduced")
        {
            if (mode != null && mode != "reduced")
            {
                throw new NotImplementedException("Only default mode='reduced' is implemented.");
            }

            return (SymbolList)_api_internal.qr(a: a, multi: true);
        }

        public _Symbol inv(_Symbol a)
        {
            return _api_internal.inv(a);
        }

        public _Symbol det(_Symbol a)
        {
            return _api_internal.det(a);
        }

        public SymbolList slogdet(_Symbol a)
        {
            return (SymbolList)_api_internal.slogdet(a: a, multi: true);
        }

        public _Symbol solve(_Symbol a, _Symbol b)
        {
            return _api_internal.solve(a, b);
        }

        public _Symbol tensorinv(_Symbol a, int ind = 2)
        {
            return _api_internal.tensorinv(a, ind);
        }

        public _Symbol tensorsolve(_Symbol a, _Symbol b, params int[] axes)
        {
            return _api_internal.tensorsolve(a, b, axes);
        }

        public _Symbol eigvals(_Symbol a)
        {
            return _api_internal.eigvals(a);
        }

        public _Symbol eigvalsh(_Symbol a, string UPLO = "L")
        {
            return _api_internal.eigvalsh(a, UPLO);
        }

        public (_Symbol, _Symbol) eig(_Symbol a)
        {
            var list = (SymbolList)_api_internal.eig(a, multi: true);
            var w = list[0];
            var v = list[1];
            return (w, v);
        }

        public (_Symbol, _Symbol) eigh(_Symbol a, string UPLO = "L")
        {
            var list = (SymbolList)_api_internal.eigh(a, multi: true);
            var w = list[0];
            var v = list[1];
            return (w, v);
        }
    }
}
