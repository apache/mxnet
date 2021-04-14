using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public class MultiArray
    {
        public static int _NDARRAY_UNSUPPORTED_INDEXING = -1;
        public static int _NDARRAY_BASIC_INDEXING = 0;
        public static int _NDARRAY_ADVANCED_INDEXING = 1;
        public static int _NDARRAY_EMPTY_TUPLE_INDEXING = 2;


        public static int _NDARRAY_NO_ZERO_DIM_BOOL_ARRAY = -1;
        public static int _NDARRAY_ZERO_DIM_BOOL_ARRAY_FALSE = 0;
        public static int _NDARRAY_ZERO_DIM_BOOL_ARRAY_TRUE = 1;
        public static int _SIGNED_INT32_UPPER_LIMIT = ((int)Math.Pow(2, 31) - 1);
        public static bool? _INT64_TENSOR_SIZE_ENABLED = null;

        public static bool _int64_enabled()
        {
            return nd_np_ops.Int64Enabled();
        }

        public static IntPtr _new_alloc_handle(Shape shape, Context ctx, bool delay_alloc, DType dtype= null)
        {
            if (dtype == null)
                dtype = np.Float32;

            return new ndarray(shape, delay_alloc, ctx, dtype).NativePtr;
        }

        public static ndarray _reshape_view(ndarray a, Shape shape)
        {
            return a.reshape(shape);
        }

        public static ndarray _as_mx_np_array(NumpyDotNet.ndarray obj, Context ctx= null, bool zero_copy= false)
        {
            var tensor = DLPack.NDArrayFromNumpy(obj, zero_copy);
            return new ndarray(tensor.data);
        }

        public static ndarray _as_mx_np_array(NumpyDotNet.ndarray[] obj, Context ctx = null, bool zero_copy = false)
        {
            throw new NotImplementedException();
        }

        public static ndarray _as_mx_np_array(Array obj, Context ctx = null, bool zero_copy = false)
        {
            return new ndarray(obj, ctx);
        }

        public static (NumpyDotNet.ndarray, Context) _as_onp_array(ndarray obj)
        {
            return (obj.AsNumpy(), obj.ctx);
        }

        public static ndarray _np_ndarray_cls(IntPtr handle, bool writable= true, StorageStype stype= StorageStype.Default)
        {
            if (stype == StorageStype.Csr)
                return new Sparse.CSRNDArray(handle);

            if (stype == StorageStype.RowSparse)
                return new Sparse.RowSparseNDArray(handle);

            return new ndarray(handle);
        }
    }
}
