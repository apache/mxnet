/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using System;
using System.Collections.Generic;
using MxNet.Interop;
using NDArrayHandle = System.IntPtr;
using mx_uint = System.UInt32;
using mx_float = System.Single;
using size_t = System.UInt64;
using MxNet.Numpy;

namespace MxNet.Sparse
{
    public class BaseSparseNDArray : ndarray
    {
        public BaseSparseNDArray()
        {

        }

        internal BaseSparseNDArray(NDArrayHandle handle)
            : base(handle)
        {

        }

        private readonly Dictionary<StorageStype, DType[]> _STORAGE_AUX_TYPES = new Dictionary<StorageStype, DType[]>
        {
            {
                StorageStype.Csr,
                new[] {DType.Int64, DType.Int64}
            },
            {
                StorageStype.RowSparse,
                new[] {DType.Int64}
            }
        };

        internal int NumAux => _STORAGE_AUX_TYPES[stype].Length;

        public override long size => throw new NotSupportedException("Not supported for Sparse NDArray");

        public override void SyncCopyFromCPU(Array data)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public override ndarray reshape(params int[] shape)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public override ndarray reshape(Shape shape, bool reverse = false)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public override ndarray Slice(int begin, int? end)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        private DType AuxType(int i)
        {
            NativeMethods.MXNDArrayGetAuxType(GetHandle(), i, out var out_type);
            return DType.GetType(out_type);
        }

        private DType[] AuxTypes()
        {
            var aux_types = new List<DType>();
            var num_aux = NumAux;
            for (var i = 0; i < num_aux; i++) aux_types.Add(AuxType(i));

            return aux_types.ToArray();
        }

        public override NumpyDotNet.ndarray AsNumpy()
        {
            return ToSType(StorageStype.Default).AsNumpy();
        }

        public override ndarray AsType(DType dtype)
        {
            return base.AsType(dtype);
        }

        public void CheckFormat(bool full_check = true)
        {
            NativeMethods.MXNDArraySyncCheckFormat(GetHandle(), full_check);
        }

        public ndarray Data()
        {
            WaitToRead();
            NativeMethods.MXNDArrayGetDataNDArray(GetHandle(), out var @out);
            return new ndarray(@out);
        }

        internal ndarray AuxData(int i)
        {
            WaitToRead();
            NativeMethods.MXNDArrayGetAuxNDArray(GetHandle(), i, out var @out);
            return new ndarray(@out);
        }

        #region Basic Ops

        public static BaseSparseNDArray operator +(BaseSparseNDArray lhs, BaseSparseNDArray rhs)
        {
            return (BaseSparseNDArray) nd.ElemwiseAdd(lhs, rhs);
        }

        public static BaseSparseNDArray operator +(BaseSparseNDArray lhs, float scalar)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator +(float scalar, BaseSparseNDArray rhs)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator -(BaseSparseNDArray lhs, BaseSparseNDArray rhs)
        {
            return (BaseSparseNDArray) nd.ElemwiseSub(lhs, rhs);
        }

        public static BaseSparseNDArray operator -(BaseSparseNDArray lhs, float scalar)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator -(float scalar, BaseSparseNDArray rhs)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator *(BaseSparseNDArray lhs, BaseSparseNDArray rhs)
        {
            return (BaseSparseNDArray) nd.ElemwiseMul(lhs, rhs);
        }

        public static BaseSparseNDArray operator *(BaseSparseNDArray lhs, float scalar)
        {
            return (BaseSparseNDArray) nd.MulScalar(lhs, scalar);
        }

        public static BaseSparseNDArray operator *(float scalar, BaseSparseNDArray rhs)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator /(BaseSparseNDArray lhs, BaseSparseNDArray rhs)
        {
            return (BaseSparseNDArray) nd.ElemwiseDiv(lhs, rhs);
        }

        public static BaseSparseNDArray operator /(BaseSparseNDArray lhs, float scalar)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        public static BaseSparseNDArray operator /(float scalar, BaseSparseNDArray rhs)
        {
            throw new NotSupportedException("Not supported for Sparse NDArray");
        }

        #endregion
    }
}