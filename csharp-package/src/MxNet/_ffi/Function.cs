using MxNet.Interop;
using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace MxNet._ffi
{
    public class Function : FunctionBase
    {
        public static (MXNetValue[], TypeCode[], int) MakeMxnetArgs(List<object> args, List<object> temp_args)
        {
            var num_args = args.Count;
            var values = new MXNetValue[num_args];
            var type_codes = new TypeCode[num_args];
            foreach (var _tup_1 in args.Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1)))
            {
                var i = _tup_1.Item1;
                var arg = _tup_1.Item2;
                if (arg is NDArray)
                {
                    values[i].v_handle = ((NDArray)arg).NativePtr;
                    type_codes[i] = TypeCode.NDARRAYHANDLE;
                }
                else if (arg is ndarray)
                {
                    values[i].v_handle = ((ndarray)arg).NativePtr;
                    type_codes[i] = TypeCode.NDARRAYHANDLE;
                }
                else if (arg is int || arg is long)
                {
                    values[i].v_int64 = (long)arg;
                    type_codes[i] = TypeCode.INT;
                }
                else if (arg is ObjectBase)
                {
                    values[i].v_handle = ((ObjectBase)arg).handle;
                    type_codes[i] = TypeCode.OBJECT_HANDLE;
                }
                else if (arg == null)
                {
                    values[i].v_handle = new IntPtr();
                    type_codes[i] = TypeCode.NULL;
                }
                else if (arg is object)
                {
                    values[i].v_handle = GCHandle.Alloc(arg).AddrOfPinnedObject();
                    type_codes[i] = TypeCode.OBJECT_HANDLE;
                }
                else if (arg is float || arg is double)
                {
                    values[i].v_float64 = (double)arg;
                    type_codes[i] = TypeCode.FLOAT;
                }
                else if (arg is string)
                {
                    unsafe
                    {
                        values[i].v_str = (char*)GCHandle.Alloc(arg).AddrOfPinnedObject();
                        type_codes[i] = TypeCode.STR;
                    }
                }
                //else if (arg is list || arg is tuple || arg is dict)
                //{
                //    arg = _FUNC_CONVERT_TO_NODE(arg);
                //    values[i].v_handle = arg.handle;
                //    type_codes[i] = TypeCode.OBJECT_HANDLE;
                //    temp_args.append(arg);
                //}
                else if (arg is IntPtr)
                {
                    values[i].v_handle = (IntPtr)arg;
                    type_codes[i] = TypeCode.HANDLE;
                }
                else if (arg is DType)
                {
                    unsafe
                    {
                        values[i].v_str = (char*)GCHandle.Alloc(((DType)arg).Name).AddrOfPinnedObject();
                        type_codes[i] = TypeCode.STR;
                    }
                }
                else
                {
                    throw new Exception($"Don't know how to handle type {arg.GetType().Name}");
                }
            }
            return (values, type_codes, num_args);
        }

        public static IntPtr InitHandleConstructor(FunctionBase fconstructor, params object[] args)
        {
            var temp_args = new List<object>();
            var (values, tcodes, num_args) = MakeMxnetArgs(args.ToList(), temp_args);
            if (NativeMethods.MXNetFuncCall(fconstructor.handle, values.Select(x => GCHandle.Alloc(x).AddrOfPinnedObject()).ToArray(), tcodes, num_args, out var ret_val, out var ret_tcode) != 0)
            {
                throw mx.GetLastFfiError();
            }

            Debug.Assert(ret_tcode == (int)TypeCode.OBJECT_HANDLE);
            var val = Marshal.PtrToStructure<MXNetValue>(ret_val);
            return val.v_handle;
        }

        public static FunctionBase MakePackedFunc(IntPtr handle, bool is_global)
        {
            var obj = new Function();
            obj.is_global = is_global;
            obj.handle = handle;
            return obj;
        }

        public static FunctionBase GetGlobalFunc(string name, bool allow_missing = false)
        {
            NativeMethods.MXNetFuncGetGlobal(name, out var handle);
            if (handle != null || handle != IntPtr.Zero)
            {
                return MakePackedFunc(handle, false);
            }

            if (allow_missing)
            {
                return null;
            }

            throw new Exception(String.Format("Cannot find global function %s", name));
        }

        public static List<string> ListGlobalFuncNames()
        {
            NativeMethods.MXNetFuncListGlobalNames(out var size, out var ptr);
            IntPtr[] listPtrs = new IntPtr[size];
            Marshal.Copy(ptr, listPtrs, 0, size);
            return listPtrs.Select(x => Marshal.PtrToStringAnsi(x)).ToList();
        }

        public static Function GetApi(Function f)
        {
            var flocal = f;
            flocal.is_global = true;
            return flocal;
        }
    }

    public class FunctionBase : IDisposable
    {
        public IntPtr handle;
        public bool is_global;

        public FunctionBase()
        {

        }

        public FunctionBase(IntPtr handle, bool is_global)
        {
            this.handle = handle;
            this.is_global = is_global;
        }

        public void Dispose()
        {
            if (!this.is_global)
            {
                NativeMethods.MXNetFuncFree(this.handle);
            }
        }

        public virtual object Call(params object[] args)
        {
            var temp_args = new List<object>();
            var (values, tcodes, num_args) = Function.MakeMxnetArgs(args.ToList(), temp_args);

            List<IntPtr> valuePtrs = new List<IntPtr>();
            foreach (var item in values)
            {
                IntPtr ptr = new IntPtr();
                Marshal.StructureToPtr<MXNetValue>(item, ptr, true);
                valuePtrs.Add(ptr);
            }

            if (NativeMethods.MXNetFuncCall(this.handle, valuePtrs.ToArray(), tcodes, num_args, out var ret_val_ptr, out var ret_tcode) != 0)
            {
                throw mx.GetLastFfiError();
            }

            var ret_val = Marshal.PtrToStructure<MXNetValue>(ret_val_ptr);

            return GetValue(ret_val, (TypeCode)ret_tcode);
        }

        public static Dictionary<TypeCode, Type> RETURN_SWITCH = new Dictionary<TypeCode, Type>()
        {
            { TypeCode.INT, typeof(long) },
            { TypeCode.FLOAT, typeof(double) },
            { TypeCode.NULL, typeof(Nullable) },
            { TypeCode.NDARRAYHANDLE, typeof(IntPtr) }
        };

        public static object GetValue(MXNetValue value, TypeCode type)
        {
            switch (type)
            {
                case TypeCode.INT:
                    return value.v_int64;
                    break;
                case TypeCode.FLOAT:
                    return value.v_float64;
                case TypeCode.HANDLE:
                    return value.v_handle;
                case TypeCode.NULL:
                    return null;
                case TypeCode.NDARRAYHANDLE:
                    return value.v_handle;
                default:
                    break;
            }

            return null;
        }
    }
}
