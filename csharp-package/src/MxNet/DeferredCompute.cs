using MxNet.Interop;
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public class DeferredCompute
    {
        public static bool IsDeferredCompute()
        {
            NativeMethods.MXNDArrayIsDeferredCompute(out var curr);
            return curr;
        }

        public static bool SetDeferredCompute(bool state)
        {
            NativeMethods.MXNDArraySetIsDeferredCompute(state, out var prev);
            return prev;
        }

        public static void Context(bool state = true)
        {
            // Like other MXNet context manager, this bleeds state across concurrent
            // code: "Context managers that have state should use Context Variables
            // instead of threading.local() to prevent their state from bleeding to
            // other code unexpectedly, when used in concurrent code."
            // https://github.com/apache/incubator-mxnet/issues/17495#issuecomment-585461965
            var val = SetDeferredCompute(state);
            try
            {
                return;
            }
            finally
            {
                SetDeferredCompute(val);
            }
        }

        public static _Symbol GetSymbol(NDArrayList output_arrays)
        {
            NativeMethods.MXNDArrayGetDeferredComputeSymbol(output_arrays.Handles, output_arrays.Length, out var handle);
            return new _Symbol(handle);
        }

        public static void SetVariable(NDArrayList arrays, SymbolList variables)
        {
            NativeMethods.MXNDArraySetDeferredComputeVariable(arrays.Handles, variables.Handles, arrays.Length);
        }
    }
}
