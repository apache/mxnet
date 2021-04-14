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
using mx_uint = System.UInt32;
using mx_float = System.Single;
using size_t = System.UInt64;

namespace MxNet.Sparse
{
    public class RowSparseNDArray : BaseSparseNDArray
    {
        public RowSparseNDArray()
        {

        }

        internal RowSparseNDArray(IntPtr handle) : base(handle)
        {
        }

        public RowSparseNDArray Indices => (RowSparseNDArray) AuxData(0);

        public new RowSparseNDArray Data => (RowSparseNDArray) Data();

        public static RowSparseNDArray operator +(RowSparseNDArray lhs, float scalar)
        {
            return (RowSparseNDArray) nd.PlusScalar(lhs, scalar);
        }

        public static RowSparseNDArray operator +(float scalar, RowSparseNDArray rhs)
        {
            return (RowSparseNDArray) nd.PlusScalar(rhs, scalar);
        }

        public static RowSparseNDArray operator -(RowSparseNDArray lhs, float scalar)
        {
            return (RowSparseNDArray) nd.MinusScalar(lhs, scalar);
        }

        public static RowSparseNDArray operator -(float scalar, RowSparseNDArray rhs)
        {
            return (RowSparseNDArray) nd.RminusScalar(rhs, scalar);
        }

        public static RowSparseNDArray operator *(RowSparseNDArray lhs, float scalar)
        {
            return (RowSparseNDArray) nd.MulScalar(lhs, scalar);
        }

        public static RowSparseNDArray operator *(float scalar, RowSparseNDArray rhs)
        {
            return (RowSparseNDArray) nd.MulScalar(rhs, scalar);
        }

        public static RowSparseNDArray operator /(RowSparseNDArray lhs, float scalar)
        {
            return (RowSparseNDArray) nd.DivScalar(lhs, scalar);
        }

        public static RowSparseNDArray operator /(float scalar, RowSparseNDArray rhs)
        {
            return (RowSparseNDArray) nd.RdivScalar(rhs, scalar);
        }

        public new void CopyTo(NDArray other)
        {
            if (other.SType == StorageStype.Csr)
                throw new Exception("CopyTo does not support destination NDArray stype Csr");
            base.CopyTo(other);
        }

        public void CopyTo(Context other)
        {
            ChangeContext(other);
        }

        public new CSRNDArray ToSType(StorageStype stype)
        {
            if (stype == StorageStype.Csr)
                throw new Exception("cast_storage from row_sparse to Csr is not supported");

            return (CSRNDArray) nd.CastStorage(this, stype);
        }
    }
}