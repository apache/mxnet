using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class GlobalVar
    {
        public static Type _ndarray_cls = null;

        public static Type _np_ndarray_cls = null;

        public static void SetNDArrayClass(Type cls)
        {
            _ndarray_cls = cls;
        }

        public static void SetNumpyNDArrayClass(Type cls)
        {
            _np_ndarray_cls = cls;
        }
    }
}
