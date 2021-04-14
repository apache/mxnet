using MxNet.Interop;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace MxNet.Numpy
{
    public partial class Utils
    {
        public static void save(string file, ndarray arr)
        {
            NDArrayList list = new NDArrayList(arr);
            List<string> keys = new List<string>();
            var handles = list.Handles;
            NativeMethods.MXNDArraySave(file, (uint)handles.Length, handles, keys.ToArray());
        }

        public static void savez(string file, NDArrayDict args)
        {
            var str_keys = args.Keys;
            var nd_vals = args.Values;
            var handles = nd_vals.Handles;
            NativeMethods.MXNDArraySave(file, (uint)handles.Length, handles, str_keys);
        }

        public static void savez(string file, NDArrayList args)
        {
            NDArrayDict dict = new NDArrayDict();
            for(int i = 0; i< args.Length; i++)
            {
                dict[$"arr_{i}"] = args[i];
            }

            savez(file, dict);
        }

        public static NDArrayDict load(string file)
        {
            if (!(MxUtil.IsNpShape() && MxUtil.IsNpArray()))
            {
                throw new Exception("Cannot load `mxnet.numpy.ndarray` in legacy mode. Please activate numpy semantics by calling `npx.set_np()` in the global scope before calling this function.");
            }

            NativeMethods.MXNDArrayLoad(file, out var out_size, out var handles, out var out_name_size, out var names);
            var outArr = new IntPtr[out_size];
            Marshal.Copy(handles, outArr, 0, (int)out_size);
            NDArrayDict data = new NDArrayDict();

            if (out_name_size == 0)
            {
                for (var i = 0; i < outArr.Length; i++) 
                    data.Add(i.ToString(), new ndarray(outArr[i]));
            }
            else
            {
                var outNames = new IntPtr[out_name_size];
                Marshal.Copy(names, outNames, 0, (int)out_name_size);

                for (var i = 0; i < outArr.Length; i++)
                {
                    var key = Marshal.PtrToStringAnsi(outNames[i]);
                    if (!string.IsNullOrEmpty(key)) data.Add(key, new ndarray(outArr[i]));
                }
            }

            return data;
        }
    }
}
