using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class DLTensor
    {
        public IntPtr data;
        public DLContext ctx;
        public int ndim;
        public DLDataType dtype;
        public IntPtr shape;
        public IntPtr strides;
        public ulong byte_offset;
    }
}
