using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet._ffi
{
    public enum TypeCode
    {
        INT = 0,
        UINT = 1,
        FLOAT = 2,
        HANDLE = 3,
        NULL = 4,
        MXNET_TYPE = 5,
        MXNET_CONTEXT = 6,
        OBJECT_HANDLE = 7,
        STR = 8,
        BYTES = 9,
        PYARG = 10,
        NDARRAYHANDLE = 11,
        EXT_BEGIN = 15
    }
}
