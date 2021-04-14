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
using MxNet.Interop;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

namespace MxNet.RTC
{
    public class CudaModule : IDisposable
    {
        public IntPtr Handle;

        private Dictionary<string, DType> cudaDtypeMap = new Dictionary<string, DType>()
        {
            {"float", DType.Float32 },
            {"double", DType.Float64 },
            {"__half", DType.Float16 },
            {"uint8_t", DType.UInt8 },
            {"int", DType.Int32 },
            {"int32_t", DType.Int32 },
            {"int8_t", DType.Int8 },
            {"char", DType.Int8 },
            {"int64_t", DType.Int64 },
        };

        public CudaModule(string source, string[] options = null, string[] exports = null)
        {
            NativeMethods.MXRtcCudaModuleCreate(source, options.Length, options, exports.Length, exports, out Handle);
        }

        public void Dispose()
        {
            if (Handle != null)
                NativeMethods.MXRtcCudaModuleFree(Handle);
        }

        public CudaKernel GetKernel(string name, string signature)
        {
            var pattern = new Regex(@"""^\s*(const)?\s*([\w_]+)\s*(\*)?\s*([\w_]+)?\s*$""");
            var args = Regex.Replace(signature, @"\s+", " ").Split(',');
            List<bool> is_const = new List<bool>();
            List<bool> is_ndarray = new List<bool>();
            List<int> dtypes = new List<int>();

            foreach (var arg in args)
            {
                var match = pattern.Match(arg);
                if (!match.Success || match.Groups[1].Value == "const")
                    throw new Exception($"Invalid function prototype \"{ arg }\". Must be in the " + 
                                         "form of \"(const) type (*) (name)\"");

                is_const.Add(match.Groups[0].Success);
                string dtype = match.Groups[1].Value;
                is_ndarray.Add(match.Groups[2].Success);
                if (!cudaDtypeMap.ContainsKey(dtype))
                    throw new Exception($"Unsupported kernel argument type {arg}");

                dtypes.Add(cudaDtypeMap[dtype].Index);
            }

            NativeMethods.MXRtcCudaKernelCreate(Handle, name, dtypes.Count, is_ndarray.ToArray(), is_const.ToArray(), dtypes.ToArray(), out var kernel_handle);

            return new CudaKernel(kernel_handle, name, is_ndarray.ToArray(), dtypes.Select(x=>(DType.GetType(x))).ToArray());
        }
    }
}