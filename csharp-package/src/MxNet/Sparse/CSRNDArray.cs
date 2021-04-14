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
using MxNet.Numpy;

namespace MxNet.Sparse
{
    public class CSRNDArray : BaseSparseNDArray
    {
        internal CSRNDArray(IntPtr handle) : base(handle)
        {
        }

        public ndarray Indices => AuxData(1);

        public ndarray IndPtr => AuxData(0);

        public new ndarray Data => base.Data();

        public static CSRNDArray operator +(CSRNDArray lhs, float scalar)
        {
            return (CSRNDArray) nd.PlusScalar(lhs, scalar);
        }

        public static CSRNDArray operator +(float scalar, CSRNDArray rhs)
        {
            return (CSRNDArray) nd.PlusScalar(rhs, scalar);
        }

        public static CSRNDArray operator -(CSRNDArray lhs, float scalar)
        {
            return (CSRNDArray) nd.MinusScalar(lhs, scalar);
        }

        public static CSRNDArray operator -(float scalar, CSRNDArray rhs)
        {
            return (CSRNDArray) nd.RminusScalar(rhs, scalar);
        }

        public static CSRNDArray operator *(CSRNDArray lhs, float scalar)
        {
            return (CSRNDArray) nd.MulScalar(lhs, scalar);
        }

        public static CSRNDArray operator *(float scalar, CSRNDArray rhs)
        {
            return (CSRNDArray) nd.MulScalar(rhs, scalar);
        }

        public static CSRNDArray operator /(CSRNDArray lhs, float scalar)
        {
            return (CSRNDArray) nd.DivScalar(lhs, scalar);
        }

        public static CSRNDArray operator /(float scalar, CSRNDArray rhs)
        {
            return (CSRNDArray) nd.RdivScalar(rhs, scalar);
        }

        public new void CopyTo(NDArray other)
        {
            if (other.SType == StorageStype.RowSparse)
                throw new Exception("CopyTo does not support destination NDArray stype RowSparse");
            base.CopyTo(other);
        }

        public void CopyTo(Context other)
        {
            ChangeContext(other);
        }

        public new CSRNDArray ToSType(StorageStype stype)
        {
            if (stype == StorageStype.RowSparse)
                throw new Exception("cast_storage from csr to row_sparse is not supported");

            return (CSRNDArray) nd.CastStorage(this, stype);
        }
    }
}