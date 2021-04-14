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
using AtomicSymbolCreator = System.IntPtr;
using DataIterCreator = System.IntPtr;
using DataIterHandle = System.IntPtr;
using ExecutorHandle = System.IntPtr;
using ExecutorMonitorCallback = System.IntPtr;
using KVStoreHandle = System.IntPtr;
using mx_uint = System.UInt32;
using NDArrayHandle = System.IntPtr;
using ProfileHandle = System.IntPtr;
using size_t = System.UInt64;
using SymbolHandle = System.IntPtr;
using CudaModuleHandle = System.IntPtr;
using CudaKernelHandle = System.IntPtr;
using uint64_t = System.UInt64;
using MxNet._ffi;

// ReSharper disable once CheckNamespace
namespace MxNet.Interop
{
    internal sealed partial class NativeMethods
    {
        #region Callbacks

        public delegate void ExecutorMonitorCallbackDelegate(string str, NDArrayHandle arrayHandle,
            ExecutorHandle executeHandle);

        #endregion

        #region Methods

        #region Part 0: Global State setups

        /// <summary>
        ///     Notify the engine about a shutdown, This can help engine to print less messages into display.
        ///     <para>User do not have to call this function.</para>
        /// </summary>
        /// <returns>0 when success, -1 when failure happens.</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNotifyShutdown();

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXStorageEmptyCache(int dev_type, int dev_id);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXGetGPUMemoryInformation64(int dev_id, ref long free_mem, ref long total_mem);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXGetGPUCount(ref int count);

        [DllImport("Kernel32.dll")]
        public static extern ExecutorHandle LoadLibrary(string path);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXGetVersion(out int version);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSetIsNumpyShape(bool is_np_shape, ref bool prev);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXIsNumpyShape(ref bool curr);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSetIsNumpyDefaultDtype(bool is_np_default_dtype, ref bool prev);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXIsNumpyDefaultDtype(ref bool curr);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXGetEnv(string name, out string value);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSetEnv(string name, string value);

        #endregion

        #region Part 1: NDArray creation and deletion

        /// <summary>
        ///     create a NDArray handle that is not initialized can be used to pass in as mutate variables to hold the result of
        ///     NDArray
        /// </summary>
        /// <param name="out">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreateNone(out NDArrayHandle @out);

        /// <summary>
        ///     free the narray handle
        /// </summary>
        /// <param name="symbol">the handle to be freed</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayFree(NDArrayHandle symbol);

        /// <summary>
        ///     get the context of the NDArray
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dev_type">the output device type</param>
        /// <param name="out_dev_id">the output device id</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetContext(NDArrayHandle handle,
            out int out_dev_type,
            out int out_dev_id);

        /// <summary>
        ///     get the content of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the ndarray</param>
        /// <param name="out_pdata">pointer holder to get pointer of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetData(NDArrayHandle handle, out AtomicSymbolCreator out_pdata);

        /// <summary>
        ///     get the content of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the ndarray</param>
        /// <param name="out_dtype">pointer holder to get pointer of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetDType(NDArrayHandle handle, out int out_dtype);

        /// <summary>
        ///     Load list of narray from the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="out_size">number of narray loaded.</param>
        /// <param name="out_arr">head of the returning narray handles.</param>
        /// <param name="out_name_size">size of output name arrray.</param>
        /// <param name="out_names">the names of returning NDArrays, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayLoad([MarshalAs(UnmanagedType.LPStr)] string fname,
            out uint out_size,
            out AtomicSymbolCreator out_arr,
            out uint out_name_size,
            out AtomicSymbolCreator out_names);

        /// <summary>
        ///     Load list / dictionary of narrays from file content loaded into memory. 
        ///     This will load a list of ndarrays in a similar manner to MXNDArrayLoad, however, it loads from buffer containing the contents of a file, rather than from a specified file.
        /// </summary>
        /// <param name="ndarray_buffer">pointer to the start of the ndarray file content</param>
        /// <param name="size">size of the file</param>
        /// <param name="out_size">number of narray loaded.</param>
        /// <param name="out_arr">head of the returning narray handles.</param>
        /// <param name="out_name_size">size of output name arrray.</param>
        /// <param name="out_names">the names of returning NDArrays, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayLoadFromBuffer(byte[] ndarray_buffer,
            int size,
            out uint out_size,
            out AtomicSymbolCreator out_arr,
            out uint out_name_size,
            out AtomicSymbolCreator out_names);

        /// <summary>
        ///     Save list of narray into the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="num_args">number of arguments to save.</param>
        /// <param name="args">the array of NDArrayHandles to be saved.</param>
        /// <param name="keys">the name of the NDArray, optional, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySave([MarshalAs(UnmanagedType.LPStr)] string fname,
            uint num_args,
            NDArrayHandle[] args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys);

        /// <summary>
        ///     get the shape of the array
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dim">the output dimension</param>
        /// <param name="out_pdata">pointer holder to get data pointer of the shape</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetShape(NDArrayHandle handle,
            out int out_dim,
            out AtomicSymbolCreator out_pdata);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetShape64(NDArrayHandle handle,
          out int out_dim,
          out AtomicSymbolCreator out_pdata);

        /// <summary>
        ///     Slice the NDArray along axis 0.
        /// </summary>
        /// <param name="handle">the handle to the NDArray</param>
        /// <param name="slice_begin">The beginning index of slice</param>
        /// <param name="slice_end">The ending index of slice</param>
        /// <param name="out">The NDArrayHandle of sliced NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySlice(NDArrayHandle handle,
            int slice_begin,
            int slice_end,
            out NDArrayHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayReshape64(NDArrayHandle handle,
            int ndim,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)]
            int[] dims,
            bool reverse,
            out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, int[] shape,
            int ndim, int dtype, out NDArrayHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetSharedMemHandle(NDArrayHandle handle, out int shared_pid,
            out int shared_id);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySaveRawBytes(NDArrayHandle handle, out int out_size,
            out AtomicSymbolCreator out_buf);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayLoadFromRawBytes(byte[] buf, int size, out NDArrayHandle handle);

        /// <summary>
        ///     Perform a synchronize copy from a continugous CPU memory region.
        ///     <para>
        ///         This function will call WaitToWrite before the copy is performed. This is useful to copy data from existing
        ///         memory region that are not wrapped by NDArray(thus dependency not being tracked).
        ///     </para>
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy from.</param>
        /// <param name="size">the memory size we want to copy from.</param>
        /// <returns></returns>
        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern int MXNDArraySyncCopyFromCPU(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.R4)] float[] data, uint size);

        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern int MXNDArraySyncCopyFromCPU(IntPtr handle, [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I4)] int[] data, uint size);
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySyncCopyFromCPU(AtomicSymbolCreator handle, AtomicSymbolCreator data,
            uint size);

        /// <summary>
        /// Perform a synchronize copyto a continugous CPU memory region.
        /// <para>This function will call WaitToRead before the copy is performed. This is useful to copy data from existing memory region that are not wrapped by NDArray(thus dependency not being tracked).</para>
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy into.</param>
        /// <param name="size">the memory size we want to copy into.</param>
        /// <returns></returns>
        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
        //                                                mx_float[] data,
        //                                                size_t size);

        /// Return Type: int
        /// handle: NDArrayHandle->void*
        /// data: void*
        /// size: size_t->unsigned int
        [DllImport(NativeLibrary, EntryPoint = "MXNDArraySyncCopyToCPU", CallingConvention = CallingConvention)]
        public static extern int MXNDArraySyncCopyToCPU(AtomicSymbolCreator handle, AtomicSymbolCreator data,
            ulong size);

        /// <summary>
        ///     wait until all delayed operations in the system is completed
        /// </summary>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitAll();

        /// <summary>
        ///     Wait until all the pending writes with respect NDArray are finished. Always call this before read data out
        ///     synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitToRead(NDArrayHandle handle);

        /// <summary>
        ///     Wait until all the pending read/write with respect NDArray are finished. Always call this before write data into
        ///     NDArray synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayWaitToWrite(NDArrayHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXLibInfoFeatures(out IntPtr features, out int size);

        #endregion

        #region Part 2: functions on NDArray

        /// <summary>
        ///     invoke a nnvm op and imperative function
        /// </summary>
        /// <param name="creator">the op</param>
        /// <param name="num_inputs">number of input NDArrays</param>
        /// <param name="inputs">input NDArrays</param>
        /// <param name="num_outputs">number of output NDArrays</param>
        /// <param name="outputs">output NDArrays</param>
        /// <param name="num_params">number of keyword parameters</param>
        /// <param name="param_keys">keys for keyword parameters</param>
        /// <param name="param_vals">values for keyword parameters</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXImperativeInvoke(AtomicSymbolCreator creator,
            int num_inputs,
            NDArrayHandle[] inputs,
            ref int num_outputs,
            ref NDArrayHandle outputs,
            int num_params,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] param_keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] param_vals);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetAuxType(AtomicSymbolCreator handle, int i, out int out_type);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetStorageType(AtomicSymbolCreator handle, out int out_storage_type);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySyncCheckFormat(AtomicSymbolCreator handle, bool full_check);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetDataNDArray(AtomicSymbolCreator handle, out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int
            MXNDArrayGetAuxNDArray(AtomicSymbolCreator handle, int i, out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayDetach(AtomicSymbolCreator handle, out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXCreateCachedOpEx(AtomicSymbolCreator handle, int num_flags,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] vals,
            out AtomicSymbolCreator @out, bool thread_safe = false);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXFreeCachedOp(AtomicSymbolCreator handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetGradState(AtomicSymbolCreator handle, out bool freshGrad);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetGrad(NDArrayHandle handle, out NDArrayHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySetGradState(AtomicSymbolCreator handle, bool freshGrad);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXInvokeCachedOp(AtomicSymbolCreator handle, int num_inputs, NDArrayHandle[] inputs,
            out int num_outputs, out NDArrayHandle[] outputs, out int[] out_stypes);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRandomSeed(int seed);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRandomSeedContext(int seed, int dev_type, int dev_id);
        #endregion

        #region Part 3: symbolic configuration generation

        /// <summary>
        ///     list all the available operator names, include entries.
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXListAllOpNames(out uint out_size, out IntPtr out_array);

        /// <summary>
        ///     This function will change the sym hanlde. To achieve function apply behavior, copy the symbol first before apply.
        /// </summary>
        /// <param name="sym">the symbol to apply</param>
        /// <param name="name">the name of symbol</param>
        /// <param name="num_args">number of arguments</param>
        /// <param name="keys">the key of keyword args(optional)</param>
        /// <param name="args">arguments to sym</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCompose(SymbolHandle sym,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            uint num_args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            SymbolHandle[] args);

        /// <summary>
        ///     Create an AtomicSymbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator</param>
        /// <param name="num_param"> the number of parameters</param>
        /// <param name="keys">keys to the params</param>
        /// <param name="vals">the vals of the params</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
            uint num_param,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] vals,
            out SymbolHandle @out);

        /// <summary>
        ///     Load a symbol from a json file.
        /// </summary>
        /// <param name="fname">the file name.</param>
        /// <param name="out">the output symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateFromFile([MarshalAs(UnmanagedType.LPStr)] string fname,
            out SymbolHandle @out);

        /// <summary>
        ///     Load a symbol from a json string.
        /// </summary>
        /// <param name="json">the json string.</param>
        /// <param name="out">the output symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateFromJSON([MarshalAs(UnmanagedType.LPStr)] string json,
            out SymbolHandle @out);

        /// <summary>
        ///     Create a Symbol by grouping list of symbols together
        /// </summary>
        /// <param name="num_symbols">number of symbols to be grouped</param>
        /// <param name="symbols">array of symbol handles</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateGroup(uint num_symbols,
            SymbolHandle[] symbols,
            out SymbolHandle @out);

        /// <summary>
        ///     Create a Variable Symbol.
        /// </summary>
        /// <param name="name">name of the variable</param>
        /// <param name="@out">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCreateVariable([MarshalAs(UnmanagedType.LPStr)] string name,
            out SymbolHandle @out);

        /// <summary>
        ///     Free the symbol handle.
        /// </summary>
        /// <param name="symbol">symbol the symbol</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolFree(SymbolHandle symbol);

        /// <summary>
        ///     Get string name from symbol
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="out">The result name.</param>
        /// <param name="success">Whether the result is contained in out.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetName(SymbolHandle symbol, out AtomicSymbolCreator @out, out int success);

        /// <summary>
        /// Get a symbol that contains all the internals.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="out">The output symbol whose outputs are all the internals.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetInternals(SymbolHandle symbol, out SymbolHandle @out);

        /// <summary>
        ///     Get a symbol that contains only direct children.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="out">The output symbol whose outputs are the direct children.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetChildren(SymbolHandle symbol, out SymbolHandle @out);

        /// <summary>
        ///     Get index-th outputs of the symbol.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="index">the Index of the output.</param>
        /// <param name="@out">The output symbol whose outputs are the index-th symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetOutput(SymbolHandle symbol,
            uint index,
            out SymbolHandle @out);

        /// <summary>
        ///     Get the detailed information about atomic symbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <param name="key_var_num_args">
        ///     The keyword argument for specifying variable number of arguments.
        ///     <para>
        ///         When this parameter has non-zero length, the function allows variable number of positional arguments, and
        ///         will need the caller to pass it in in MXSymbolCreateAtomicSymbol, With key = key_var_num_args, and value =
        ///         number of positional arguments.
        ///     </para>
        /// </param>
        /// <param name="return_type">Return type of the function, can be Symbol or SymbolList</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
            out AtomicSymbolCreator name,
            out AtomicSymbolCreator description,
            out uint num_args,
            out AtomicSymbolCreator arg_names,
            out AtomicSymbolCreator arg_type_infos,
            out AtomicSymbolCreator arg_descriptions,
            out AtomicSymbolCreator key_var_num_args,
            ref AtomicSymbolCreator return_type);

        /// <summary>
        ///     Get the detailed information about atomic symbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <param name="key_var_num_args">
        ///     The keyword argument for specifying variable number of arguments.
        ///     <para>
        ///         When this parameter has non-zero length, the function allows variable number of positional arguments, and
        ///         will need the caller to pass it in in MXSymbolCreateAtomicSymbol, With key = key_var_num_args, and value =
        ///         number of positional arguments.
        ///     </para>
        /// </param>
        /// <param name="return_type">Return type of the function, can be Symbol or SymbolList</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
            out AtomicSymbolCreator name,
            out AtomicSymbolCreator description,
            out uint num_args,
            out AtomicSymbolCreator[] arg_names,
            out AtomicSymbolCreator[] arg_type_infos,
            out AtomicSymbolCreator[] arg_descriptions,
            out AtomicSymbolCreator key_var_num_args,
            out AtomicSymbolCreator return_type);

        /// <summary>
        ///     infer shape of unknown input shapes given the known one.
        ///     <para>The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data</para>
        ///     <para>The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.</para>
        /// </summary>
        /// <param name="sym">symbol handle</param>
        /// <param name="num_args">numbe of input arguments.</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="arg_ind_ptr">the head pointer of the rows in CSR</param>
        /// <param name="arg_shape_data">the content of the CSR</param>
        /// <param name="in_shape_size">sizeof the returning array of in_shapes</param>
        /// <param name="in_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="in_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="out_shape_size">sizeof the returning array of out_shapes</param>
        /// <param name="out_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="out_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="aux_shape_size">sizeof the returning array of aux_shapes</param>
        /// <param name="aux_shape_ndim">returning array of shape dimensions of eachs auxiliary shape.</param>
        /// <param name="aux_shape_data">returning array of pointers to head of the auxiliary shape.</param>
        /// <param name="complete">whether infer shape completes or more information is needed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXSymbolInferShape(SymbolHandle sym,
            uint num_args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            int* in_shape_size,
            int** in_shape_ndim,
            int*** in_shape_data,
            out int out_shape_size,
            out int* out_shape_ndim,
            out int** out_shape_data,
            out int aux_shape_size,
            out int* aux_shape_ndim,
            out int** aux_shape_data,
            out int complete);

        //[DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        //public static extern unsafe int MXSymbolInferShape(SymbolHandle sym,
        //    uint num_args,
        //    [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
        //    string[] keys,
        //    int[] arg_ind_ptr,
        //    int[] arg_shape_data,
        //    int* in_shape_size,
        //    int** in_shape_ndim,
        //    int*** in_shape_data,
        //    out int out_shape_size,
        //    out int* out_shape_ndim,
        //    out int** out_shape_data,
        //    out int aux_shape_size,
        //    out int* aux_shape_ndim,
        //    out int** aux_shape_data,
        //    out int complete);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXSymbolInferShapePartial(SymbolHandle sym,
            uint num_args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            int[] arg_ind_ptr,
            int[] arg_shape_data,
            int* in_shape_size,
            int** in_shape_ndim,
            int*** in_shape_data,
            out int out_shape_size,
            out int* out_shape_ndim,
            out int** out_shape_data,
            out int aux_shape_size,
            out int* aux_shape_ndim,
            out int** aux_shape_data,
            out int complete);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXSymbolInferType(SymbolHandle sym,
            uint num_args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            int[] arg_type_data,
            int* in_type_size,
            int** in_type_data,
            out int out_type_size,
            out int* out_type_data,
            out int aux_type_size,
            out int* aux_type_data,
            out int complete);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXSymbolInferTypePartial(SymbolHandle sym,
            uint num_args,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            int[] arg_type_data,
            int* in_type_size,
            int** in_type_data,
            out int out_type_size,
            out int* out_type_data,
            out int aux_type_size,
            out int* aux_type_data,
            out int complete);

        /// <summary>
        ///     List arguments in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListArguments(SymbolHandle symbol,
            out uint out_size,
            out AtomicSymbolCreator out_str_array);

        /// <summary>
        ///     list all the available AtomicSymbolEntry
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output AtomicSymbolCreator array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAtomicSymbolCreators(out uint out_size, out AtomicSymbolCreator out_array);

        /// <summary>
        ///     List auxiliary states in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
            out uint out_size,
            out AtomicSymbolCreator out_str_array);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAttr(SymbolHandle symbol,
          out uint out_size,
          out AtomicSymbolCreator out_str_array);

        /// <summary>
        ///     List returns in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListOutputs(SymbolHandle symbol,
            out uint out_size,
            out AtomicSymbolCreator out_str_array);

        /// <summary>
        ///     Save a symbol into a json file.
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="fname">the file name.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolSaveToFile(SymbolHandle symbol, [MarshalAs(UnmanagedType.LPStr)] string fname);

        /// <summary>
        ///     Save a symbol into a json string
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="out_json">output json string.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolSaveToJSON(SymbolHandle symbol, out AtomicSymbolCreator out_json);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolRemoveAmpCast(SymbolHandle symbol, out SymbolHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetAttr(SymbolHandle symbol, string key, out string @out, out int success);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolSetAttr(SymbolHandle symbol, string key, string value);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolListAttrShallow(SymbolHandle symbol, out int out_size,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] out
            string[] @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolCompose(SymbolHandle symbol, string name, int num_args,
           [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] keys, SymbolHandle[] args);

        #endregion

        #region Part 4: Executor interface

        /// <summary>
        ///     Excecutor run backward
        /// </summary>
        /// <param name="handle">execute handle</param>
        /// <param name="len">lenth</param>
        /// <param name="head_grads">NDArray handle for heads' gradient</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorBackward(ExecutorHandle handle, uint len, NDArrayHandle[] head_grads);

        /*!
         * \brief Generate Executor from symbol,
         *  This is advanced function, allow specify group2ctx map.
         *  The user can annotate "ctx_group" attribute to name each group.
         *
         * \param symbol_handle symbol handle
         * \param dev_type device type of default context
         * \param dev_id device id of default context
         * \param num_map_keys size of group2ctx map
         * \param map_keys keys of group2ctx map
         * \param map_dev_types device type of group2ctx map
         * \param map_dev_ids device id of group2ctx map
         * \param len length
         * \param in_args in args array
         * \param arg_grad_store arg grads handle array
         * \param grad_req_type grad req array
         * \param aux_states_len length of auxiliary states
         * \param aux_states auxiliary states array
         * \param shared_exec input executor handle for memory sharing
         * \param out output executor handle
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorBindEX(SymbolHandle symbol_handle,
            int dev_type,
            int dev_id,
            uint num_map_keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] map_keys,
            int[] map_dev_types,
            int[] map_dev_ids,
            uint len,
            NDArrayHandle[] in_args,
            NDArrayHandle[] arg_grad_store,
            uint[] grad_req_type,
            uint aux_states_len,
            NDArrayHandle[] aux_states,
            ExecutorHandle shared_exec,
            out ExecutorHandle @out);

        /// <summary>
        ///     Executor forward method
        /// </summary>
        /// <param name="handle">executor handle</param>
        /// <param name="is_train">int value to indicate whether the forward pass is for evaluation</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorForward(ExecutorHandle handle, int is_train);

        /// <summary>
        ///     Delete the executor
        /// </summary>
        /// <param name="handle">the executor.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorFree(ExecutorHandle handle);

        /// <summary>
        ///     Get executor's head NDArray
        /// </summary>
        /// <param name="handle">executor handle</param>
        /// <param name="out_size">output narray vector size</param>
        /// <param name="out">out put narray handles</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorOutputs(ExecutorHandle handle, out uint out_size,
            out AtomicSymbolCreator @out);

        /// <summary>
        ///     Print the content of execution plan, used for debug.
        /// </summary>
        /// <param name="handle">the executor.</param>
        /// <param name="out_str">pointer to hold the output string of the printing.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorPrint(ExecutorHandle handle, out AtomicSymbolCreator out_str);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern unsafe int MXExecutorSimpleBindEx(SymbolHandle symbol_handle,
                                   int dev_type,
                                   int dev_id,
                                   int num_g2c_keys,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] g2c_keys,
                                   int[] g2c_dev_types,
                                   int[] g2c_dev_ids,
                                   int provided_grad_req_list_len,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] provided_grad_req_names,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] provided_grad_req_types,
                                   int num_provided_arg_shapes,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] provided_arg_shape_names,
                                   int[] provided_arg_shape_data,
                                   int[] provided_arg_shape_idx,
                                   int num_provided_arg_dtypes,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] provided_arg_dtype_names,
                                   int[] provided_arg_dtypes,
                                   int num_provided_arg_stypes,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] provided_arg_stype_names,
                                   int[] provided_arg_stypes,
                                   int num_shared_arg_names,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] shared_arg_name_list,
                                   int* shared_buffer_len,
                                   [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] shared_buffer_name_list,
                                   NDArrayHandle[] shared_buffer_handle_list,
                                   char*** updated_shared_buffer_name_list,
                                   NDArrayHandle** updated_shared_buffer_handle_list,
                                   int* num_in_args,
                                   NDArrayHandle** in_args,
                                   NDArrayHandle** arg_grads,
                                   int* num_aux_states,
                                   NDArrayHandle** aux_states,
                                   ExecutorHandle shared_exec_handle,
                                   ExecutorHandle* @out);

        #endregion

        #region Part 5: IO Interface

        /// <summary>
        ///     set a call back to notify the completion of operation
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="callback"></param>
        /// <param name="callback_handle"></param>
        /// <returns></returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXExecutorSetMonitorCallback(ExecutorHandle handle,
            ExecutorMonitorCallback callback,
            AtomicSymbolCreator callback_handle);

        /// <summary>
        ///     List all the available iterator entries
        /// </summary>
        /// <param name="out_size">the size of returned iterators</param>
        /// <param name="out_array">the output iteratos entries</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXListDataIters(out uint out_size, out AtomicSymbolCreator out_array);

        /// <summary>
        ///     Init an iterator, init with parameters the array size of passed in arguments
        /// </summary>
        /// <param name="handle">handle of the iterator creator</param>
        /// <param name="num_param">number of parameter</param>
        /// <param name="keys">parameter keys</param>
        /// <param name="vals">parameter values</param>
        /// <param name="out">resulting iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterCreateIter(DataIterCreator handle,
            uint num_param,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] vals,
            out DataIterHandle @out);

        /// <summary>
        ///     Get the detailed information about data iterator.
        /// </summary>
        /// <param name="creator">the DataIterCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names">Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments.</param>
        /// <param name="arg_descriptions">Description information about the arguments.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetIterInfo(DataIterCreator creator,
            out AtomicSymbolCreator name,
            out AtomicSymbolCreator description,
            out uint num_args,
            out AtomicSymbolCreator arg_names,
            out AtomicSymbolCreator arg_type_infos,
            out AtomicSymbolCreator arg_descriptions);

        /// <summary>
        ///     Free the handle to the IO module
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterFree(DataIterHandle handle);

        /// <summary>
        ///     Move iterator to next position
        /// </summary>
        /// <param name="handle">the handle to iterator</param>
        /// <param name="out">return value of next</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterNext(DataIterHandle handle, out int? @out);

        /// <summary>
        ///     Call iterator.Reset
        /// </summary>
        /// <param name="handle">the handle to iterator</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterBeforeFirst(DataIterHandle handle);

        /// <summary>
        ///     Get the handle to the NDArray of underlying data
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out">handle to underlying data NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetData(DataIterHandle handle, out NDArrayHandle @out);

        /// <summary>
        ///     Get the image index by array.
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out_index">output index of the array.</param>
        /// <param name="out_size">output size of the array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetIndex(DataIterHandle handle,
            out AtomicSymbolCreator out_index,
            out ulong out_size);

        /// <summary>
        ///     Get the padding number in current data batch
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="pad">pad number ptr</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetPadNum(DataIterHandle handle, out int pad);

        /// <summary>
        ///     Get the handle to the NDArray of underlying label
        /// </summary>
        /// <param name="handle">the handle pointer to the data iterator</param>
        /// <param name="out">the handle to underlying label NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetLabel(DataIterHandle handle, out NDArrayHandle @out);

        #endregion

        #region Part 6: advanced KVStore for multi-machines

        /// <summary>
        ///     create a NDArray with specified shape
        /// </summary>
        /// <param name="shape">the pointer to the shape</param>
        /// <param name="ndim">the dimension of the shape</param>
        /// <param name="dev_type">device type, specify device we want to take</param>
        /// <param name="dev_id">the device id of the specific device</param>
        /// <param name="delay_alloc">whether to delay allocation until the narray is first mutated</param>
        /// <param name="@out">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreate(
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)]
            int[] shape,
            int ndim, DeviceType devType,
            int devId,
            int delayAlloc,
            int dtype,
            out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayCreate64(
           [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)]
            int[] shape,
           int ndim, DeviceType devType,
           int devId,
           int delayAlloc,
           int dtype,
           out AtomicSymbolCreator @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreCreate(string type, out KVStoreHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreGetRank(KVStoreHandle handle, out int rank);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreGetGroupSize(KVStoreHandle handle, out int size);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreGetType(KVStoreHandle handle,
            [Out] [MarshalAs(UnmanagedType.LPStr)] string type);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreRunServer(KVStoreHandle handle, MXKVStoreServerController controller,
            AtomicSymbolCreator controller_handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreSendCommmandToServers(KVStoreHandle handle, int cmd_id, string cmd_body);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreIsWorkerNode(out int ret);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreFree(KVStoreHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreInitEx(KVStoreHandle handle, int num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys, NDArrayHandle[] vals);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreInit(KVStoreHandle handle, int num, int[] keys, NDArrayHandle[] vals);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePushEx(KVStoreHandle handle, int num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            NDArrayHandle[] vals, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePush(KVStoreHandle handle, int num,
            int[] keys, NDArrayHandle[] vals, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePullWithSparseEx(KVStoreHandle handle, int num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            NDArrayHandle[] vals, int priority, bool ignore_sparse);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePullWithSparse(KVStoreHandle handle, int num,
            int[] keys, NDArrayHandle[] vals, int priority, bool ignore_sparse);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePushPullEx(KVStoreHandle handle, int vnum,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] vkeys,
            int onum,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] okeys,
            NDArrayHandle[] vals, NDArrayHandle[] outs, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePushPull(KVStoreHandle handle, int vnum,
            int[] vkeys, int onum, int[] okeys,
            NDArrayHandle[] vals, NDArrayHandle[] outs, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePullRowSparseEx(KVStoreHandle handle, int num,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            NDArrayHandle[] vals, NDArrayHandle[] row_ids, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStorePullRowSparse(KVStoreHandle handle, int num,
            int[] keys, NDArrayHandle[] vals, NDArrayHandle[] row_ids, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreSetGradientCompression(KVStoreHandle handle, int num_params,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] keys,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)]
            string[] vals);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreSetUpdaterEx(KVStoreHandle handle, AtomicSymbolCreator updater,
            AtomicSymbolCreator str_updater, AtomicSymbolCreator updater_handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreBarrier(KVStoreHandle handle);

        #endregion

        #region Part 7: Autograd API

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradSetIsRecording(int is_recording, ref int prev);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradSetIsTraining(int train_mode, ref int prev);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradIsRecording(ref int curr);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradIsTraining(ref int curr);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradMarkVariables(int num_var, NDArrayHandle[] var_handles,
            [In] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)]
            int[] reqs_array, NDArrayHandle[] grad_handles);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradBackwardEx(int num_output, NDArrayHandle[] output_handles,
            NDArrayHandle[] ograd_handles, int num_variables,
            NDArrayHandle[] var_handles, int retain_graph, int create_graph, int is_train,
            out NDArrayHandle[] grad_handles,
            [Out] [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.SysInt)]
            out int[] grad_stypes);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle @out);

        #endregion

        #region Engine API

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXEngineSetBulkSize(int size, ref int prev);

        #endregion

        #region Profiler API

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileCreateDomain(string name, out ProfileHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileCreateTask(ProfileHandle handle, string name, out ProfileHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileCreateFrame(ProfileHandle handle, string name, out ProfileHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileCreateEvent(string name, out ProfileHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileCreateCounter(ProfileHandle handle, string name, out ProfileHandle @out);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileDestroyHandle(ProfileHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileDurationStart(ProfileHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileDurationStop(ProfileHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileSetCounter(ProfileHandle handle, int value);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileAdjustCounter(ProfileHandle handle, int delta);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXProfileSetMarker(ProfileHandle handle, string name, string scope);

        #endregion

        #region Cuda Module API
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRtcCudaModuleCreate(string source, int num_options, [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options, int num_exports, [In][MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] exports, out CudaModuleHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRtcCudaModuleFree(CudaModuleHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRtcCudaKernelFree(CudaKernelHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRtcCudaKernelCreate(CudaModuleHandle handle, string name, int num_args, bool[] is_ndarray, bool[] is_const, int[] arg_types, out CudaKernelHandle 
                    kernelHandle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRtcCudaKernelCall(CudaKernelHandle handle, int dev_id, IntPtr[] args, int grid_dim_x, int grid_dim_y, int grid_dim_z, int block_dim_x, int block_dim_y, int block_dim_z, int shared_mem);
        #endregion

        #region RecordIO API
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOWriterCreate(string uri, out IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOReaderCreate(string uri, out IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOWriterFree(IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOReaderFree(IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOWriterWriteRecord(IntPtr handle, byte[] buff, int size);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOReaderReadRecord(IntPtr handle, out IntPtr buff_ptr, out int size);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOReaderSeek(IntPtr handle, int pos);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXRecordIOWriterTell(IntPtr handle, out int pos);
        #endregion


        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayIsDeferredCompute(out bool ret);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySetIsDeferredCompute(bool state, out bool prev);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArrayGetDeferredComputeSymbol(NDArrayHandle[] output_handles, int count, out SymbolHandle handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNDArraySetDeferredComputeVariable(NDArrayHandle[] arrays, NDArrayHandle[] variables, int count);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreBroadcastEx(NDArrayHandle handle, int nCvkeys, string[] cvKeys, int nCoKeys, string[] coKeys, NDArrayHandle[] cvals, NDArrayHandle[] couts, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXKVStoreBroadcast(NDArrayHandle handle, int nCvkeys, int[] cvKeys, int nCoKeys, int[] coKeys, NDArrayHandle[] cvals, NDArrayHandle[] couts, int priority);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetLenHint(NDArrayHandle handle, out int length);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXDataIterGetItems(NDArrayHandle handle, out int num_output, NDArrayHandle[] output_vars);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXShallowCopySymbol(SymbolHandle handle, out SymbolHandle hdl);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSymbolGetInputs(SymbolHandle handle, out SymbolHandle hdl);
        
        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXSetProfilerScope(string name);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNetFuncFree(IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNetFuncCall(IntPtr handle, IntPtr[] values, _ffi.TypeCode[] type_codes, int num_args, out IntPtr ret_value, out int ret_tcode);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNetObjectGetTypeIndex(IntPtr handle, out int index);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNetFuncGetGlobal(string name, out IntPtr handle);

        [DllImport(NativeLibrary, CallingConvention = CallingConvention)]
        public static extern int MXNetFuncListGlobalNames(out int size, out IntPtr handle);

        #endregion
    }
}