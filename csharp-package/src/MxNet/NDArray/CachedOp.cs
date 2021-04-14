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
using System.Linq;
using MxNet.Interop;
using CachedOpHandle = System.IntPtr;
using mx_uint = System.UInt32;
using mx_float = System.Single;
using size_t = System.UInt64;
using System.Collections.Generic;
using MxNet.Numpy;
using MxNet.Sym.Numpy;

namespace MxNet
{
    public class CachedOp : IDisposable
    {
        private readonly CachedOpHandle handle;

        public CachedOp(_Symbol sym, IDictionary<string, string> flags = null, bool thread_safe = false)
        {
            handle = IntPtr.Zero;
            if (flags == null)
                flags = new Dictionary<string, string>();
            Logging.CHECK_EQ(NativeMethods.MXCreateCachedOpEx(sym.GetHandle(), flags.Count, flags.Keys.ToArray(),
                flags.Values.ToArray(), out handle, thread_safe), NativeMethods.OK);
        }

        public _Symbol GetOptimizedSymbol()
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            if (handle != null)
                NativeMethods.MXFreeCachedOp(handle);
        }

        public NDArrayList Call(NDArrayList args)
        {
            Logging.CHECK_EQ(NativeMethods.MXInvokeCachedOp(handle, args.Length, MxUtil.GetNDArrayHandles(args), out var num_outputs,
                out var outputs, out var out_stypes), NativeMethods.OK);
            var result = new NDArrayList();
            for (var i = 0; i < num_outputs; i++)
                result.Add(new ndarray(outputs[i]).ToSType((StorageStype) out_stypes[i]));

            return result.ToArray();
        }

        public void RegisteropHook(Action<string, string, ndarray> callback, bool monitor_all = false)
        {
            throw new NotImplementedException();
        }
    }
}