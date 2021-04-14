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
using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MxNet
{
    public static class ObjectExtensions
    {
        public static string ToValueString(this object source)
        {
            if (source is bool)
                return ((bool)source) ? "1" : "0";
            else if(source is Array)
            {
                string result = "";
                var arr = (Array)source;
                result = "[";
                List<string> localList = new List<string>();
                for (int i = 0; i < arr.Length; i++)
                {
                    localList.Add(arr.GetValue(i).ToString());
                }

                result += string.Join(",", localList);
                result += "]";

                return result;
            }

            return source.ToString();
        }

        public static IntPtr GetMemPtr(this object src)
        {
            if (src.GetType().Name == "NDArray")
                return ((NDArray)src).NativePtr;

            if (src.GetType().Name == "Symbol")
                return ((Symbol)src).NativePtr;


            GCHandle handle = GCHandle.Alloc(src, GCHandleType.Pinned);
            IntPtr pointer = handle.AddrOfPinnedObject();
            handle.Free();
            return pointer;
        }
    }
}