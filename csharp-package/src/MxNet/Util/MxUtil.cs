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
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Threading;

namespace MxNet
{
    public class _NumpyShapeScope : MxDisposable
    {
        public bool? _enter_is_np_shape;

        public bool? _prev_is_np_shape;

        public _NumpyShapeScope(bool is_np_shape)
        {
            this._enter_is_np_shape = is_np_shape;
            this._prev_is_np_shape = null;
        }



        public override void Exit()
        {
            if (this._enter_is_np_shape != null && this._prev_is_np_shape != this._enter_is_np_shape)
            {
                MxUtil.SetNpShape(this._prev_is_np_shape.Value);
            }
        }

        public override MxDisposable With()
        {
            if (this._enter_is_np_shape != null)
            {
                this._prev_is_np_shape = MxUtil.SetNpShape(this._enter_is_np_shape.Value);
            }

            return this;
        }
    }

    public class _NumpyArrayScope : MxDisposable
    {
        public bool _is_np_array;

        public _NumpyArrayScope _old_scope;

        public static ThreadLocal<_NumpyArrayScope> _current = new ThreadLocal<_NumpyArrayScope>();

        public _NumpyArrayScope(bool is_np_array)
        {
            this._old_scope = null;
            this._is_np_array = is_np_array;
        }

        public override void Exit()
        {
            Debug.Assert(this._old_scope != null);
            _current.Value = this._old_scope;
        }

        public override MxDisposable With()
        {
            this._old_scope = _current.Value;
            _current.Value = this;
            return this;
        }
    }

    public class _NumpyDefaultDtypeScope : MxDisposable
    {
        public bool? _enter_is_np_default_dtype;

        public bool? _prev_is_np_default_dtype;

        public _NumpyDefaultDtypeScope(bool is_np_default_dtype)
        {
            this._enter_is_np_default_dtype = is_np_default_dtype;
            this._prev_is_np_default_dtype = null;
        }

        public override void Exit()
        {
            if (this._enter_is_np_default_dtype != null && this._prev_is_np_default_dtype != this._enter_is_np_default_dtype)
            {
                MxUtil.SetNpDefaultDtype(this._prev_is_np_default_dtype.Value);
            }
        }

        public override MxDisposable With()
        {
            if (this._enter_is_np_default_dtype != null)
            {
                this._prev_is_np_default_dtype = MxUtil.SetNpDefaultDtype(this._enter_is_np_default_dtype.Value);
            }

            return this;
        }
    }

    public class MxUtil
    {
        public static bool _set_np_shape_logged = false;

        public static bool _set_np_array_logged = false;

        public static bool _set_np_default_dtype_logged = false;

        public static string EnumToString<TEnum>(TEnum? _enum, List<string> convert) where TEnum : struct, IConvertible
        {
            if (_enum.HasValue)
            {
                var v = _enum.Value as object;
                return convert[(int) v];
            }

            return null;
        }

        public static void ValidateParam(string name, string value, params string[] validValues)
        {
            if (!validValues.Contains(value))
            {
                var message = "Invalid value for " + name + ". Valid values are " + string.Join(", ", validValues);
                throw new Exception(message);
            }
        }

        public static IntPtr[] GetNDArrayHandles(NDArrayList list)
        {
            return list.Select(x => x.GetHandle()).ToArray();
        }

        public static List<T> Set<T>(List<T> keys)
        {
            return keys.Distinct().OrderBy(x => x).ToList();
        }

        public static (Shape, Shape) GetSliceNotation(string slice, Shape shape)
        {
            string[] split = slice.Split(',');
            int[] begin = new int[split.Length];
            int[] end = new int[split.Length];
            for (int i = 0; i < split.Length; i++)
            {
                begin[i] = 0;
                end[i] = shape[i];
                var range = split[i].Contains(":") ? split[i].Split(':') : null;
                if (range != null)
                {
                    begin[i] = !string.IsNullOrEmpty(range[0]) ? Convert.ToInt32(range[0].Trim()) : begin[i];
                    end[i] = !string.IsNullOrEmpty(range[1]) ? Convert.ToInt32(range[1].Trim()) : end[i];
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(split[i]))
                    {
                        begin[i] = Convert.ToInt32(split[i].Trim());
                        end[i] = begin[i] + 1;
                    }
                }
            }

            return (new Shape(begin), new Shape(end));
        }

        public static int GetGPUCount()
        {
            int count = 0;
            NativeMethods.MXGetGPUCount(ref count);
            return count;
        }

        public static (long, long) GetGPUMemory(int gpu_dev_id)
        {
            long free_mem = 0;
            long total_mem = 0;
            NativeMethods.MXGetGPUMemoryInformation64(gpu_dev_id, ref free_mem, ref total_mem);
            return (free_mem, total_mem);
        }

        public static bool SetNpShape(bool active)
        {
            if (active)
            {
                if (!_set_np_shape_logged)
                {
                    Logger.Info("NumPy-shape semantics has been activated in your code. This is required for creating and manipulating scalar and zero-size tensors, which were not supported in MXNet before, as in the official NumPy library. Please DO NOT manually deactivate this semantics while using `mxnet.numpy` and `mxnet.numpy_extension` modules.");
                    _set_np_shape_logged = true;
                }
            }
            else if (IsNpArray())
            {
                throw new Exception("Deactivating NumPy shape semantics while NumPy array semantics is still active is not allowed. Please consider calling `npx.reset_np()` to deactivate both of them.");
            }

            bool prev = false;
            NativeMethods.MXSetIsNumpyShape(active, ref prev);
            return prev;
        }

        public static bool IsNpShape()
        {
            bool curr = false;
            NativeMethods.MXIsNumpyShape(ref curr);
            return curr;
        }

        public static _NumpyShapeScope NpShape(bool active)
        {
            return new _NumpyShapeScope(active);
        }

        public static void SanityCheckParams(string func_name, string[] unsupported_params, Dictionary<string, object> param_dict)
        {
            foreach (var param_name in unsupported_params)
            {
                if (param_dict.ContainsKey(param_name))
                {
                    throw new NotImplementedException($"function {func_name} does not support parameter {param_name}");
                }
            }
        }

        public static _NumpyArrayScope NpArray(bool active)
        {
            return new _NumpyArrayScope(active);
        }

        public static bool IsNpArray()
        {
            if (_NumpyArrayScope._current.Value != null)
                return _NumpyArrayScope._current.Value._is_np_array;

            return false;
        }

        public static bool UseUFuncLegalOption(string key, object value)
        {
            if (key == "where")
            {
                return true;
            }
            else if (key == "casting")
            {
                return new HashSet<object>(new List<string> {
                "no",
                "equiv",
                "safe",
                "same_kind",
                "unsafe"
            }).Contains(value);
            }
            else if (key == "order")
            {
                if (value is string)
                {
                    return true;
                }
            }
            else if (key == "dtype")
            {
                return new HashSet<object>(new List<object> {
                np.Int8,
                np.UInt8,
                np.Int32,
                np.Int64,
                np.Float16,
                np.Float32,
                np.Float64,
                "int8",
                "uint8",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64"
            }).Contains(value);
            }
            else if (key == "subok")
            {
                return value is bool;
            }
            return false;
        }

        public static bool SetNpArray(bool active)
        {
            if (active)
            {
                if (!_set_np_array_logged)
                {
                    Logger.Info("NumPy array semantics has been activated in your code. This allows you to use operators from MXNet NumPy and NumPy Extension modules as well as MXNet NumPy `ndarray`s.");
                    _set_np_array_logged = true;
                }
            }
            var cur_state = IsNpArray();
            _NumpyArrayScope._current.Value = new _NumpyArrayScope(active);
            return cur_state;
        }

        public static void SetNp(bool shape = true, bool array = true, bool dtype = false)
        {
            if (!shape && array)
            {
                throw new Exception("NumPy Shape semantics is required in using NumPy array semantics.");
            }

            SetNpArray(array);
            SetNpShape(shape);
            SetNpDefaultDtype(dtype);
        }

        public static void ResetNp()
        {
            SetNp(false, false, false);
        }

        public static int GetCudaComputeCapability(Context ctx)
        {
            if (ctx.GetDeviceType() !=  DeviceType.GPU)
            {
                throw new Exception($"Expecting a gpu context to get cuda compute capability, while received ctx {ctx}");
            }

            string error_str = "";
            int version = 0;
            ProcessStartInfo startInfo = new ProcessStartInfo("nvcc", "-Vq");
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            var p = Process.Start(startInfo);
            
            p.WaitForExit();
            if (p.ExitCode == 0)
            {
                var versionData = p.StandardOutput.ReadToEnd().Split('\n');
                foreach (var item in versionData)
                {
                    if (item.Contains("Cuda compilation tools"))
                    {
                        var line = item.Replace("Cuda compilation tools, release", "");
                        line = line.Split(',')[0];
                        version = Convert.ToInt32(Convert.ToSingle(line) * 10);
                    }
                }
            }
            else
            {
                throw new Exception("Unable to find cuda: " + p.StandardError.ReadToEnd());
            }

            return version;
        }

        public static ndarray DefaultArray(Array source_array, Context ctx= null, DType dtype= null)
        {
            if (IsNpArray())
            {
                return np.array(source_array, ctx: ctx, dtype: dtype);
            }
            else
            {
                return nd.Array(source_array, ctx: ctx).Cast(dtype);
            }
        }

        public static _NumpyDefaultDtypeScope NpDefaultDtype(bool active = true)
        {
            return new _NumpyDefaultDtypeScope(active);
        }

        public static bool IsNpDefaultDtype()
        {
            bool curr = false;
            NativeMethods.MXIsNumpyDefaultDtype(ref curr);
            return curr;
        }

        public static bool SetNpDefaultDtype(bool is_np_default_dtype)
        {
            if (is_np_default_dtype)
            {
                if (!_set_np_default_dtype_logged)
                {
                    Logger.Info("NumPy array default dtype has been changed from flaot32 to float64 in your code.");
                    _set_np_default_dtype_logged = true;
                }
            }
            var prev = false;
            NativeMethods.MXSetIsNumpyDefaultDtype(is_np_default_dtype, ref prev);
            return prev;
        }

        public static string GetEnv(string name)
        {
            NativeMethods.MXGetEnv(name, out var ret);
            return ret;
        }

        public static void SetEnv(string name, string value)
        {
            NativeMethods.MXSetEnv(name, value);
        }

    }
}