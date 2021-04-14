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
using OpHandle = System.IntPtr;
using nn_uint = System.UInt32;
using SymbolHandle = System.IntPtr;

// ReSharper disable once CheckNamespace
namespace MxNet.Interop
{
    internal sealed partial class NativeMethods
    {
        #region Methods

        /// <summary>
        ///     return str message of the last error
        ///     <para>
        ///         all function in this file will return 0 when success and -1 when an error occured,
        ///         <see cref="NNGetLastError" /> can be called to retrieve the error
        ///     </para>
        ///     <para>this function is threadsafe and can be called by different thread</para>
        /// </summary>
        /// <returns>error info</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern OpHandle NNGetLastError();

        /// <summary>
        ///     Get operator handle given name.
        /// </summary>
        /// <param name="op_name">The name of the operator.</param>
        /// <param name="op_out">The returnning op handle.</param>
        /// <returns></returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int NNGetOpHandle(OpHandle op_name, out OpHandle op_out);

        /// <summary>
        ///     list all the available operator names, include entries.
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int NNListAllOpNames(out uint out_size, out OpHandle out_array);

        /// <summary>
        ///     List input names in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="option">The option to list the inputs
        ///     option=0 means list all arguments.
        ///     option=1 means list arguments that are readed only by the graph.
        ///     option=2 means list arguments that are mutated by the graph.</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>return 0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int NNSymbolListInputNames(SymbolHandle symbol,
            int type,
            out uint out_size,
            out IntPtr out_str_array);

        #endregion
    }
}