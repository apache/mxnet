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
using System.Runtime.InteropServices;

// ReSharper disable once CheckNamespace
namespace MxNet.Interop
{
    internal sealed partial class NativeMethods
    {
        #region Methods

        [DllImport(CLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr memcpy(uint[] dest, IntPtr src, uint count);

        [DllImport(CLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr memcpy(float[] dest, IntPtr src, uint count);

        #endregion
    }
}