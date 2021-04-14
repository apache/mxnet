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
using System.Runtime.InteropServices;

namespace MxNet.Interop
{
    internal sealed partial class NativeMethods
    {
        #region Constants

        public const int OK = 0;

        public const int Error = -1;

        public const int TRUE = 1;

        public const int FALSE = 0;

        #endregion

#if LINUX
        public const string NativeLibrary = "libmxnet.so";

        public const string CLibrary = "libc.so";

        public const CallingConvention CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl;
#else
        public const string NativeLibrary = @"libmxnet";

        public const string CLibrary = "msvcrt.dll";

        public const CallingConvention CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl;
#endif
    }
}