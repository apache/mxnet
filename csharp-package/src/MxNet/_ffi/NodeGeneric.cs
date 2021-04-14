using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet._ffi
{
    public class NodeGeneric
    {
        public static DType ScalarTypeInference(object value)
        {
            DType dtype = null;
            if (value.GetType().Name == "NDArray")
                dtype = ((NDArray)value).DataType;
            if (value.GetType().Name == "NDArrayOrSymbol")
                dtype = ((NDArrayOrSymbol)value).IsNDArray ? ((NDArrayOrSymbol)value).NdX.dtype : null;
            else if (value.GetType().Name == "Single")
                dtype = DType.Float32;
            else if (value.GetType().Name == "Int32")
                dtype = DType.Int32;

            return dtype;
        }

        public static dynamic ConvertToNode(string value)
        {
            throw new NotImplementedException();
        }

        public static dynamic Const(int value, DType dtype = null)
        {
            throw new NotImplementedException();
        }

        public static dynamic Const(float value, DType dtype = null)
        {
            throw new NotImplementedException();
        }
    }
}
