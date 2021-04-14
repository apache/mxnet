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
using mx_float = System.Single;
using mx_uint = System.UInt32;


// ReSharper disable once CheckNamespace
namespace MxNet.Interop
{
    internal sealed partial class NativeMethods
    {
        #region Methods

        /// <summary>
        ///     Get the last error happeneed.
        /// </summary>
        /// <returns>The last error happened at the predictor.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern IntPtr MXGetLastError();

        /// <summary>
        ///     Create a NDArray List by loading from ndarray file. This can be used to load mean image file.
        /// </summary>
        /// <param name="nd_file_bytes">The byte contents of nd file to be loaded.</param>
        /// <param name="nd_file_size">The size of the nd file to be loaded.</param>
        /// <param name="out">The out put NDListHandle</param>
        /// <param name="out_length">Length of the list.</param>
        /// <returns>return 0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDListCreate(byte[] nd_file_bytes,
            int nd_file_size,
            [Out] out IntPtr @out,
            [Out] out uint out_length);

        /// <summary>
        ///     Free a MXAPINDList
        /// </summary>
        /// <param name="handle">The handle of the MXAPINDList</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDListFree(IntPtr handle);

        /// <summary>
        ///     Get an element from list
        /// </summary>
        /// <param name="handle">The handle to the NDArray</param>
        /// <param name="index">The index in the list</param>
        /// <param name="out_key">The output key of the item</param>
        /// <param name="out_data">The data region of the item</param>
        /// <param name="out_shape">The shape of the item.</param>
        /// <param name="out_ndim">The number of dimension in the shape.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDListGet(IntPtr handle,
            uint index,
            out IntPtr out_key,
            out IntPtr out_data,
            out IntPtr out_shape,
            out uint out_ndim);

        /// <summary>
        ///     create a predictor
        /// </summary>
        /// <param name="symbol_json_str">create a predictor</param>
        /// <param name="param_bytes">The JSON string of the symbol.</param>
        /// <param name="param_size">The in-memory raw bytes of parameter ndarray file.</param>
        /// <param name="dev_type">The size of parameter ndarray file.</param>
        /// <param name="dev_id">The device type, 1: cpu, 2:gpu</param>
        /// <param name="num_input_nodes">The device id of the predictor.</param>
        /// <param name="input_keys">Number of input nodes to the net, For feedforward net, this is 1.</param>
        /// <param name="input_shape_indptr">The name of input argument. For feedforward net, this is {"data"}</param>
        /// <param name="input_shape_data">
        ///     Index pointer of shapes of each input node. For feedforward net that takes 4 dimensional
        ///     input, this is the shape data.
        /// </param>
        /// <param name="out">The created predictor handle.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredCreate([MarshalAs(UnmanagedType.LPStr)] string symbol_json_str,
            byte[] param_bytes,
            int param_size,
            int dev_type,
            int dev_id,
            uint num_input_nodes,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] input_keys,
            uint[] input_shape_indptr,
            uint[] input_shape_data,
            out IntPtr @out);

        /// <summary>
        ///     create a predictor wich customized outputs
        /// </summary>
        /// <param name="symbol_json_str">The JSON string of the symbol.</param>
        /// <param name="param_bytes">The in-memory raw bytes of parameter ndarray file.</param>
        /// <param name="param_size">The size of parameter ndarray file.</param>
        /// <param name="dev_type">The device type, 1: cpu, 2:gpu</param>
        /// <param name="dev_id">The device id of the predictor.</param>
        /// <param name="num_input_nodes">Number of input nodes to the net, For feedforward net, this is 1.</param>
        /// <param name="input_keys">The name of input argument. For feedforward net, this is {"data"}</param>
        /// <param name="input_shape_indptr">
        ///     Index pointer of shapes of each input node. The length of this array = num_input_nodes
        ///     + 1. For feedforward net that takes 4 dimensional input, this is {0, 4}.
        /// </param>
        /// <param name="input_shape_data">
        ///     A flatted data of shapes of each input node. For feedforward net that takes 4
        ///     dimensional input, this is the shape data.
        /// </param>
        /// <param name="num_output_nodes">Number of output nodes to the net,</param>
        /// <param name="output_keys">The name of output argument. For example {"global_pool"}</param>
        /// <param name="out">The created predictor handle.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredCreatePartialOut([MarshalAs(UnmanagedType.LPStr)] string symbol_json_str,
            byte[] param_bytes,
            int param_size,
            int dev_type,
            int dev_id,
            uint num_input_nodes,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] input_keys,
            uint[] input_shape_indptr,
            uint[] input_shape_data,
            uint num_output_nodes,
            string[] output_keys,
            out IntPtr @out);

        /// <summary>
        ///     Run a forward pass to get the output.
        /// </summary>
        /// <param name="handle">The handle of the predictor.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredForward(IntPtr handle);

        /// <summary>
        ///     Free a predictor handle.
        /// </summary>
        /// <param name="handle">The handle of the predictor.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredFree(IntPtr handle);

        /// <summary>
        ///     Get the output value of prediction.
        /// </summary>
        /// <param name="handle">The handle of the predictor.</param>
        /// <param name="index">The index of output node, set to 0 if there is only one output.</param>
        /// <param name="data">User allocated data to hold the output.</param>
        /// <param name="size">The size of data array, used for safe checking.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredGetOutput(IntPtr handle,
            uint index,
            float[] data,
            uint size);

        /// <summary>
        ///     Get the shape of output node. The returned shape_data and shape_ndim is only valid before next call to MXPred
        ///     function.
        /// </summary>
        /// <param name="handle">The handle of the predictor.</param>
        /// <param name="index">The index of output node, set to 0 if there is only one output.</param>
        /// <param name="shape_data">Used to hold pointer to the shape data</param>
        /// <param name="shape_ndim">Used to hold shape dimension.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredGetOutputShape(IntPtr handle,
            uint index,
            out IntPtr shape_data,
            out uint shape_ndim);

        /// <summary>
        ///     Run a interactive forward pass to get the output. This is helpful for displaying progress of prediction which can
        ///     be slow. User must call PartialForward from step=0, keep increasing it until step_left=0.
        /// </summary>
        /// <param name="handle">The handle of the predictor.</param>
        /// <param name="step">The current step to run forward on.</param>
        /// <param name="step_left">The number of steps left</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredPartialForward(IntPtr handle, int step, out int step_left);

        /// <summary>
        ///     Set the input data of predictor.
        /// </summary>
        /// <param name="handle">The predictor handle.</param>
        /// <param name="key">The name of input node to set. For feedforward net, this is "data".</param>
        /// <param name="data">The pointer to the data to be set, with the shape specified in MXPredCreate.</param>
        /// <param name="size">The size of data array, used for safety check.</param>
        /// <returns>0 when success, -1 when failure.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXPredSetInput(IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string key,
            float[] data,
            uint size);

        #endregion
    }
}