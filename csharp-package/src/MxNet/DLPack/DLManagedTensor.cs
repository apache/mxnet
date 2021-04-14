using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public struct DLManagedTensor
    {
        public byte dl_tensor;
        public IntPtr manager_ctx;
        public Action<IntPtr> deleter;
    }
}
