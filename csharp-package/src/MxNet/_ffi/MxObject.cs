using MxNet.Interop;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet._ffi
{
    public class MxObject
    {
        public Dictionary<int, Type> OBJECT_TYPE = new Dictionary<int, Type>();

        public Type _CLASS_OBJECT;

        public void SetClassObject(Type object_class)
        {
            _CLASS_OBJECT = object_class;
        }

        public void RegisterObject(int index, Type cls)
        {
            OBJECT_TYPE[index] = cls;
        }

        public object ReturnObject(MXNetValue x)
        {
            object obj;
            var handle = x.v_handle;
            NativeMethods.MXNetObjectGetTypeIndex(handle, out var tindex);
            var cls = OBJECT_TYPE.ContainsKey(tindex) ? OBJECT_TYPE[tindex] : _CLASS_OBJECT;

            obj = cls.GetConstructors()[0].Invoke(new object[0]);
            cls.GetField("handle").SetValue(obj, handle);
            return obj;
        }
    }

    public class ObjectBase : MxObject, IDisposable
    {
        public IntPtr handle;

        public void Dispose()
        {
            NativeMethods.MXNetFuncFree(this.handle);
        }

        public void InitHandleByConstructor(Function fconstructor, params object[] args)
        {
            this.handle = IntPtr.Zero;
            this.handle = Function.InitHandleConstructor(fconstructor, args);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if(obj is ObjectBase)
            {
                if (this.handle == null)
                {
                    return ((ObjectBase)obj).handle == null;
                }

                return this.handle == ((ObjectBase)obj).handle;
            }

            return false;
        }
    }
}
