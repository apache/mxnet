using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MxNet._ffi
{
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct MXNetValue
    {
        public long v_int64;
        public double v_float64;
        public IntPtr v_handle;
        public char* v_str;
        public int v_type;
    }
}
