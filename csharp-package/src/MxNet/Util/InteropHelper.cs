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

// ReSharper disable once CheckNamespace
namespace MxNet
{
    internal static class InteropHelper
    {
        #region Methods

        public static IntPtr[] ToPointerArray(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new IntPtr[count];
                var p = (void**) ptr;
                for (var i = 0; i < count; i++)
                    array[i] = (IntPtr) p[i];

                return array;
            }
        }

        public static float[] ToFloatArray(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new float[count];
                var p = (float*) ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        public static int[] ToInt32Array(IntPtr ptr, int count)
        {
            unsafe
            {
                var array = new int[count];
                var p = (int*) ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        public static long[] ToInt64Array(IntPtr ptr, int count)
        {
            unsafe
            {
                var array = new long[count];
                var p = (int*)ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        public static ulong[] ToUInt64Array(IntPtr ptr, uint count)
        {
            unsafe
            {
                var array = new ulong[count];
                var p = (ulong*) ptr;
                for (var i = 0; i < count; i++)
                    array[i] = p[i];

                return array;
            }
        }

        #endregion
    }
}