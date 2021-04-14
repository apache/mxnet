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
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MxNet.Interop;
using NDArrayHandle = System.IntPtr;
using mx_uint = System.UInt32;
using mx_float = System.Single;
using size_t = System.UInt64;
using System.IO.Compression;
using System.IO;
using System.Diagnostics;
using MxNet.ND.Numpy;
using MxNet.Numpy;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    [Obsolete("Legacy API after MxNet v2, will be deprecated in v3", false)]
    public partial class NDArray : DisposableMXNetObject
    {
        #region Fields

        internal NDBlob _Blob;

        public Context Context
        {
            get
            {
                return GetContext();
            }
        }

        public Shape Shape => new Shape(GetShape().ToArray());

        public DType DataType => DType.GetType(GetDType());

        public StorageStype SType => (StorageStype) StorageType();

        public NDArray T
        {
            get
            {
                return Transpose();
            }
        }

        #endregion

        #region Constructors

        public NDArray()
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArrayCreateNone(out var @out), NativeMethods.OK);

            NativePtr = @out;
            _Blob = new NDBlob(@out);
        }

        internal NDArray(NDArrayHandle handle)
        {
            if (handle == NDArrayHandle.Zero)
                throw new ArgumentException("Can not pass IntPtr.Zero", nameof(handle));

            NativePtr = handle;
            _Blob = new NDBlob(handle);
        }

        public NDArray(Shape shape, bool delayAlloc = true, Context ctx = null, DType dtype = null)
        {
            if (ctx == null)
                ctx = Context.CurrentContext;

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            if (dtype == null)
                dtype = DType.Float32;

            if (nd_np_ops.Int64Enabled())
            {
                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate64(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out var @out), NativeMethods.OK);
                NativePtr = @out;
                _Blob = new NDBlob(@out);
            }
            else
            {
                if (shape.Size > Int32.MaxValue)
                {
                    throw new Exception("[_new_alloc_handle] Size of tensor you are trying to allocate is " + "larger than 2^31 elements. Please build with flag " + "USE_INT64_TENSOR_SIZE=1");
                }

                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out var @out), NativeMethods.OK);
                NativePtr = @out;
                _Blob = new NDBlob(@out);
            }

            
        }

        public NDArray(Array data, Shape shape, Context ctx = null, DType dtype = null)
        {
            if (ctx == null)
                ctx = Context.CurrentContext;
            
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));
            NDArrayHandle @out = new NDArrayHandle();
            if (dtype == null)
                dtype = DType.InferDtype(data);

            if (nd_np_ops.Int64Enabled())
            {
                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate64(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out @out), NativeMethods.OK);
            }
            else
            {
                if (shape.Size > Int32.MaxValue)
                {
                    throw new Exception("[_new_alloc_handle] Size of tensor you are trying to allocate is " + "larger than 2^31 elements. Please build with flag " + "USE_INT64_TENSOR_SIZE=1");
                }

                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out @out), NativeMethods.OK);
            }

            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyFromCPU(@out, datagch.AddrOfPinnedObject(), (uint) shape.Size);
            datagch.Free();

            NativePtr = @out;
            _Blob = new NDBlob(@out);
        }

        public NDArray(Array data, Context ctx = null, DType dtype = null)
        {
            if (ctx == null)
                ctx = Context.CurrentContext;

            var shapeData = new List<int>();

            for (int i = 0; i < data.Rank; i++)
            {
                shapeData.Add(data.GetLength(i));
            }

            var shape = new Shape(shapeData.ToArray());

            if(dtype == null)
                dtype = DType.InferDtype(data);

            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            NDArrayHandle @out = new NDArrayHandle();
            if (nd_np_ops.Int64Enabled())
            {
                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate64(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out @out), NativeMethods.OK);
            }
            else
            {
                if (shape.Size > Int32.MaxValue)
                {
                    throw new Exception("[_new_alloc_handle] Size of tensor you are trying to allocate is " + "larger than 2^31 elements. Please build with flag " + "USE_INT64_TENSOR_SIZE=1");
                }

                Logging.CHECK_EQ(NativeMethods.MXNDArrayCreate(shape.Data.ToArray(),
                       shape.Dimension,
                       ctx.GetDeviceType(),
                       ctx.GetDeviceId(),
                       false.ToInt32(),
                       dtype.Index,
                       out @out), NativeMethods.OK);
            }

            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyFromCPU(@out, datagch.AddrOfPinnedObject(), (uint)shape.Size);
            datagch.Free();

            NativePtr = @out;
            _Blob = new NDBlob(@out);
        }

        #endregion

        #region Properties

        public virtual long Size
        {
            get
            {
                long ret = 1;
                var shape = GetShape();
                for (var i = 0; i < shape.Count; i++)
                    ret *= shape[i];

                return ret;
            }
        }

        public int Dimension => Shape.Dimension;

        public bool FreshGrad
        {
            get
            {
                NativeMethods.MXNDArrayGetGradState(NativePtr, out var freshGrad);
                return freshGrad;
            }
            set => NativeMethods.MXNDArraySetGradState(NativePtr, value);
        }

        public Array ArrayData => AsArray();

        public NDArray Grad
        {
            get
            {
                NativeMethods.MXNDArrayGetGrad(NativePtr, out var h);
                return new NDArray(h);
            }
        }

        public ndarray NP
        {
            get
            {
                return AsNumpy();
            }
        }

        #endregion

        #region Methods

        public NDArray Copy()
        {
            var ret = new NDArray(Shape);
            using (var op = new Operator("_copyto"))
            {
                op.Set(this).Invoke(ret);
            }

            return ret;
        }

        public NDArray CopyTo(NDArray other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            using (var op = new Operator("_copyto"))
            {
                op.Set(this).Invoke(other);
            }

            return other;
        }

        public NDArray ChangeContext(Context ctx)
        {
            var result = new NDArray(Shape, true, ctx, DataType);
            CopyTo(result);
            return result;
        }

        public Context GetContext()
        {
            NativeMethods.MXNDArrayGetContext(NativePtr, out var out_dev_type, out var out_dev_id);
            return new Context((DeviceType) out_dev_type, out_dev_id);
        }

        public NDArrayHandle GetData()
        {
            NativeMethods.MXNDArrayGetData(NativePtr, out var ret);
            if (GetDType() != 0)
                return IntPtr.Zero;

            return ret;
        }

        public int GetDType()
        {
            NativeMethods.MXNDArrayGetDType(NativePtr, out var ret);
            return ret;
        }

        public NDArrayHandle GetHandle()
        {
            ThrowIfDisposed();
            return NativePtr;
        }

        public IList<int> GetShape()
        {
            if (nd_np_ops.Int64Enabled())
            {
                NativeMethods.MXNDArrayGetShape(NativePtr, out var outDim, out var outData);
                return InteropHelper.ToInt32Array(outData, outDim);
            }
            else
            {
                NativeMethods.MXNDArrayGetShape64(NativePtr, out var outDim, out var outData);
                return InteropHelper.ToInt64Array(outData, outDim).Select(x => Convert.ToInt32(x)).ToList();
            }
        }

        public static NDArrayDict LoadToMap(string fileName)
        {
            var arrayMap = new NDArrayDict();
            Logging.CHECK_EQ(NativeMethods.MXNDArrayLoad(fileName,
                out var outSize,
                out var outArr,
                out var outNameSize,
                out var outNames), NativeMethods.OK);
            if (outNameSize > 0)
            {
                var array = InteropHelper.ToPointerArray(outArr, outSize);
                var namearray = InteropHelper.ToPointerArray(outNames, outNameSize);

                Logging.CHECK_EQ(outNameSize, outSize);
                for (uint i = 0; i < outSize; ++i)
                {
                    var name = Marshal.PtrToStringAnsi(namearray[i]);
                    arrayMap[name] = new NDArray(array[i]);
                }
            }

            return arrayMap;
        }

        public static void Save(string fileName, NDArrayDict arrayMap)
        {
            var tmp = arrayMap.Keys.ToArray();

            var args = new NDArrayHandle[tmp.Length];
            var keys = new string[tmp.Length];

            var i = 0;
            foreach (var item in arrayMap)
            {
                args[i] = item.Value.GetHandle();
                keys[i] = item.Key;
                i++;
                ;
            }

            //for (var i = 0; i < tmp.Length; i++)
            //{
            //    var kv = arrayMap[keys[i]];
            //    args[i] = kv.GetHandle();
            //    keys[i] = keys[i];
            //}

            Logging.CHECK_EQ(NativeMethods.MXNDArraySave(fileName, (uint) args.Length, args, keys), NativeMethods.OK);
        }

        public static void Save(string fileName, NDArrayList arrayList)
        {
            var args = arrayList.Select(array => array.GetHandle()).ToArray();
            Logging.CHECK_EQ(NativeMethods.MXNDArraySave(fileName, (uint) args.Length, args, null), NativeMethods.OK);
        }

        public byte[] GetBuffer()
        {
            NativeMethods.MXNDArraySaveRawBytes(NativePtr, out var out_size, out var buffPtr);
            var buff = new byte[out_size];
            Marshal.Copy(buffPtr, buff, 0, out_size);
            return buff;
        }

        public static void Load(string filename, out NDArrayDict data)
        {
            data = new NDArrayDict();
            uint outSize;
            IntPtr outArrPtr;
            uint outNameSize;
            IntPtr outNamesPtr;

            NativeMethods.MXNDArrayLoad(filename, out outSize, out outArrPtr, out outNameSize, out outNamesPtr);
            var outArr = new NDArrayHandle[outSize];
            Marshal.Copy(outArrPtr, outArr, 0, (int) outSize);


            if (outNameSize == 0)
            {
                for (var i = 0; i < outArr.Length; i++) data.Add(i.ToString(), new NDArray(outArr[i]));
            }
            else
            {
                var outNames = new IntPtr[outNameSize];
                Marshal.Copy(outNamesPtr, outNames, 0, (int) outNameSize);

                for (var i = 0; i < outArr.Length; i++)
                {
                    var key = Marshal.PtrToStringAnsi(outNames[i]);
                    if (!string.IsNullOrEmpty(key)) data.Add(key, new NDArray(outArr[i]));
                }
            }
        }

        public static NDArrayDict Load(string filename)
        {
            Load(filename, out var r);
            return r;
        }

        public static NDArray LoadFromBuffer(byte[] buffer)
        {
            NativeMethods.MXNDArrayLoadFromRawBytes(buffer, buffer.Length, out var handle);
            return new NDArray(handle);
        }

        public static NDArray LoadCV2Mat(OpenCvSharp.Mat img, Context context = null)
        {
            if (context == null)
                context = mx.Cpu();
            context = mx.Cpu();
            OpenCvSharp.Cv2.CvtColor(img, img, OpenCvSharp.ColorConversionCodes.BGR2RGB);
            Shape s = new Shape(img.Height, img.Width, img.Channels());
            byte[] bytes = new byte[s.Size];
            unsafe
            {
                Buffer.MemoryCopy(img.Data.ToPointer(), bytes.GetMemPtr().ToPointer(), bytes.Length, bytes.Length);
            }

            var ret = new NDArray(bytes, s, context, dtype: DType.UInt8);
            WaitAll();
            return ret;
        }

        public static NDArrayList LoadNpz(string file)
        {
            NDArrayList result = new NDArrayList();
            using (ZipArchive zip = ZipFile.OpenRead(file))
            {
                foreach (ZipArchiveEntry entry in zip.Entries)
                {
                    Stream fs = entry.Open();
                    BinaryReader reader = new BinaryReader(fs);
                    var magic = reader.ReadChars(6);
                    var maj = reader.ReadByte();
                    var min = reader.ReadByte();
                    int headerLength = reader.ReadUInt16();
                    string header = new string(reader.ReadChars(headerLength)).Trim();
                    string mark = "'descr': '";
                    int s = header.IndexOf(mark) + mark.Length;
                    int e = header.IndexOf("'", s + 1);
                    string type = header.Substring(s, e - s);

                    DType dtype = GetNpyType(type);
                    mark = "'fortran_order': ";
                    s = header.IndexOf(mark) + mark.Length;
                    e = header.IndexOf(",", s + 1);
                    bool fortran = bool.Parse(header.Substring(s, e - s));

                    if (fortran)
                        throw new Exception();

                    mark = "'shape': (";
                    s = header.IndexOf(mark) + mark.Length;
                    e = header.IndexOf(")", s + 1);
                    var shapeSplit = header.Substring(s, e - s).Split(',');
                    List<int> shapeInt = new List<int>();
                    foreach (var element in shapeSplit)
                    {
                        if (!string.IsNullOrWhiteSpace(element))
                        {
                            shapeInt.Add(Convert.ToInt32(element));
                        }
                    }

                    Shape shape = new Shape(shapeInt);
                    if (dtype == DType.Int32)
                    {
                        List<int> data = new List<int>();
                        for (int i = 0; i < shape.Size; i++)
                        {
                            data.Add(reader.ReadInt32());
                        }

                        var x = nd.Array(data.ToArray()).AsType(dtype).Reshape(shape);
                        result.Add(x);
                    }
                    else if (dtype == DType.Int8)
                    {
                        List<sbyte> data = new List<sbyte>();
                        for (int i = 0; i < shape.Size; i++)
                        {
                            data.Add(reader.ReadSByte());
                        }

                        var x = nd.Array(data.ToArray()).AsType(dtype).Reshape(shape);
                        result.Add(x);
                    }
                    else if (dtype == DType.Int64)
                    {
                        List<long> data = new List<long>();
                        for (int i = 0; i < shape.Size; i++)
                        {
                            data.Add(reader.ReadSByte());
                        }

                        var x = nd.Array(data.ToArray()).AsType(dtype).Reshape(shape);
                        result.Add(x);
                    }
                    else if (dtype == DType.Float32)
                    {
                        List<float> data = new List<float>();
                        for (int i = 0; i < shape.Size; i++)
                        {
                            data.Add(reader.ReadSByte());
                        }

                        var x = nd.Array(data.ToArray()).AsType(dtype).Reshape(shape);
                        result.Add(x);
                    }
                    else if (dtype == DType.Float64)
                    {
                        List<double> data = new List<double>();
                        for (int i = 0; i < shape.Size; i++)
                        {
                            data.Add(reader.ReadSByte());
                        }

                        var x = nd.Array(data.Select(i => (float)i).ToArray()).AsType(dtype).Reshape(shape).AsType(dtype);
                        result.Add(x);
                    }
                    else if (dtype == DType.UInt8)
                    {
                        var data = reader.ReadBytes(Convert.ToInt32(shape.Size));

                        var x = nd.Array(data.Select(i => (float)i).ToArray()).Reshape(shape).AsType(dtype);
                        result.Add(x);
                    }
                }
            }

            return result;
        }

        private static DType GetNpyType(string dtype)
        {
            string typeCode = dtype.Substring(1);

            if (typeCode == "i1")
                return DType.Int8;
            if (typeCode == "u1")
                return DType.UInt8;
            if (typeCode == "i2")
                return DType.Int32;
            if (typeCode == "i4")
                return DType.Int32;
            if (typeCode == "i8")
                return DType.Int64;
            if (typeCode == "f4")
                return DType.Float32;
            if (typeCode == "f8")
                return DType.Float64;

            throw new NotSupportedException();
        }

        public static NDArray NewFromSharedMem(int shared_pid, int shared_id, Shape shape, DType dtype)
        {
            NativeMethods.MXNDArrayCreateFromSharedMem(shared_pid, shared_id, shape.Data.ToArray(), shape.Dimension,
                dtype.Index, out var handle);
            return new NDArray(handle);
        }

        public (int, int, Shape, DType) ToSharedMem()
        {
            NativeMethods.MXNDArrayGetSharedMemHandle(NativePtr, out var shared_pid, out var shared_id);
            return (shared_pid, shared_id, Shape, DataType);
        }

        public void Constant(float scalar)
        {
            using (var op = new Operator("_set_value"))
            {
                op.Set(scalar).Invoke(this);
            }
        }

        public NDArray SliceAxis(int axis, int begin, int? end)
        {
            var @out = new NDArray();
            if(end.HasValue)
                new Operator("slice_axis")
                    .SetParam("axis", axis)
                    .SetParam("begin", begin)
                    .SetParam("end", end.Value)
                    .SetInput("data", this)
                    .Invoke(@out);
            else
                new Operator("slice_axis")
                .SetParam("axis", axis)
                .SetParam("begin", begin)
                .SetParam("end", "None")
                .SetInput("data", this)
                .Invoke(@out);

            return @out;
        }

        public virtual NDArray Slice(int begin, int? end)
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArraySlice(GetHandle(), begin, end.Value, out var handle),
                NativeMethods.OK);
            return new NDArray(handle);
        }

        public NDArray SliceAssignScalar(double value, Shape begin, Shape end, Shape step)
        {
            return nd.SliceAssignScalar(this, scalar: value, begin: begin, end: end, step: step);
        }

        public NDArray SliceAssign(NDArray rhs, Shape begin, Shape end, Shape step)
        {
            return nd.SliceAssign(this, rhs: rhs, begin: begin, end: end, step: step);
        }

        public void SyncCopyFromCPU(Array data, ulong size)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var resize = size > 0;
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);

            NativeMethods.MXNDArraySyncCopyFromCPU(NativePtr, datagch.AddrOfPinnedObject(), (uint) size);

            datagch.Free();
        }

        public virtual void SyncCopyFromCPU(Array data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyFromCPU(NativePtr, datagch.AddrOfPinnedObject(), (uint) data.Length);
            datagch.Free();
        }

        public void SyncCopyToCPU(Array data)
        {
            SyncCopyToCPU(data, 0);
        }

        public void SyncCopyToCPU(Array data, int size = 0)
        {
            var resize = size > 0;
            size = resize ? size : GetShape().Count();
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyToCPU(NativePtr, datagch.AddrOfPinnedObject(), (ulong) size);

            datagch.Free();
        }

        public void SampleGaussian(float mu = 0, float sigma = 1)
        {
            using (var op = new Operator("_random_normal"))
            {
                op.Set(mu, sigma).Invoke(this);
            }
        }

        public void SampleUniform(float low = 0f, float high = 1f)
        {
            using (var op = new Operator("_random_uniform"))
            {
                op.Set(low, high).Invoke(this);
            }
        }

        public Array AsArray()
        {
            var size = Size;
            Array data = null;
            
            switch (DataType.Name)
            {
                case "float16":
                    data = Array.CreateInstance(typeof(float), size);
                    break;
                case "float32":
                    data = Array.CreateInstance(typeof(float), size);
                    break;
                case "float64":
                    data = Array.CreateInstance(typeof(double), size);
                    break;
                case "int8":
                    data = Array.CreateInstance(typeof(byte), size);
                    break;
                case "uint8":
                    data = Array.CreateInstance(typeof(sbyte), size);
                    break;
                case "int32":
                    data = Array.CreateInstance(typeof(int), size);
                    break;
                case "int64":
                    data = Array.CreateInstance(typeof(long), size);
                    break;
            }

            GCHandle datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyToCPU(NativePtr, datagch.AddrOfPinnedObject(), (ulong) size);
            datagch.Free();
            return data;
        }

        public T[] GetValues<T>()
        {
            return AsArray().OfType<T>().ToArray();
        }

        public T AsScalar<T>()
        {
            return (T)Convert.ChangeType(AsArray().GetValue(0), typeof(T));
        }

        public static void WaitAll()
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArrayWaitAll(), NativeMethods.OK);
        }

        public void WaitToRead()
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArrayWaitToRead(NativePtr), NativeMethods.OK);
        }

        public void WaitToWrite()
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArrayWaitToWrite(NativePtr), NativeMethods.OK);
        }

        public virtual NDArray AsType(DType dtype)
        {
            return nd.Cast(this, dtype);
        }

        public NDArray AsInContext(Context context)
        {
            if (this.Context == context)
                return this;

            return ChangeContext(context);
        }

        private int StorageType()
        {
            NativeMethods.MXNDArrayGetStorageType(GetHandle(), out var out_storage_type);
            return out_storage_type;
        }

        public virtual ndarray AsNumpy()
        {
            return new ndarray(this.NativePtr);
        }

        public NDArray this[int index]
        {
            get
            {
                var x = Slice(index, index + 1);
                var new_shape = x.Shape.Data.ToList();
                new_shape.RemoveAt(0);
                return x.Reshape(new Shape(new_shape));
            }
        }

        public NDArray this[string slice]
        {
            get
            {
                if (string.IsNullOrEmpty(slice))
                    return this;

                var (begin, end) = MxUtil.GetSliceNotation(slice, Shape);

                return Slice(begin, end);
            }
            set
            {
                if (string.IsNullOrEmpty(slice))
                    value.CopyTo(this);

                var (begin, end) = MxUtil.GetSliceNotation(slice, Shape);
                NDArray output = null;

                if (value.Size == 1)
                    output = nd.SliceAssignScalar(this, begin, end, value.AsScalar<double>());
                else
                    output = nd.SliceAssign(this, value, begin, end);

                output.CopyTo(this);
            }
        }

        public NDArray this[NDArray indices] => nd.Take(this, indices);

        public void AttachGrad(OpGradReq grad_req = OpGradReq.Write, StorageStype? stype = null)
        {
            NDArray grad = nd.ZerosLike(this);
            if (stype.HasValue)
                grad = grad.ToSType(stype.Value);

            Autograd.MarkVariables(NP, grad.NP, grad_req);
        }

        public NDArray Detach()
        {
            NativeMethods.MXNDArrayDetach(GetHandle(), out var hdl);
            return new NDArray(hdl);
        }

        public void Backward(NDArray out_grad = null, bool retain_graph = false, bool train_mode = true)
        {
            var ograd_handles = new List<NDArrayHandle>();
            var var_handles = new List<NDArrayHandle>();
            //var grad_handles = new List<NDArrayHandle>();
            if (out_grad != null)
                ograd_handles.Add(out_grad.GetHandle());
            else
                ograd_handles.Add(new NDArrayHandle());

            NativeMethods.MXAutogradBackwardEx(1, new NDArrayHandle[1] {NativePtr}, ograd_handles.ToArray(),
                0, var_handles.ToArray(), retain_graph ? 1 : 0,
                0, train_mode ? 1 : 0, out var grad_handles, out var grad_count);
        }

        #region Operators

        public static NDArray operator +(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastAdd(lhs, rhs);
        }

        public static NDArray operator +(NDArray lhs, float scalar)
        {
            return nd.PlusScalar(lhs, scalar);
        }

        public static NDArray operator +(float scalar, NDArray rhs)
        {
            return nd.PlusScalar(rhs, scalar);
        }

        public static NDArray operator -(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastSub(lhs, rhs);
        }

        public static NDArray operator -(NDArray lhs, float scalar)
        {
            return nd.MinusScalar(lhs, scalar);
        }

        public static NDArray operator -(float scalar, NDArray rhs)
        {
            return nd.RminusScalar(rhs, scalar);
        }

        public static NDArray operator *(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastMul(lhs, rhs);
        }

        public static NDArray operator *(NDArray lhs, float scalar)
        {
            return nd.MulScalar(lhs, scalar);
        }

        public static NDArray operator *(float scalar, NDArray rhs)
        {
            return nd.MulScalar(rhs, scalar);
        }

        public static NDArray operator /(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastDiv(lhs, rhs);
        }

        public static NDArray operator /(NDArray lhs, float scalar)
        {
            return nd.DivScalar(lhs, scalar);
        }

        public static NDArray operator /(float scalar, NDArray rhs)
        {
            return nd.RdivScalar(rhs, scalar);
        }

        public static NDArray operator %(NDArray lhs, float scalar)
        {
            var ret = new NDArray();
            using (var op = new Operator("_mod_scalar"))
            {
                op.Set(lhs, scalar).Invoke(ret);
            }

            return ret;
        }

        public static NDArray operator %(NDArray lhs, NDArray rhs)
        {
            var ret = new NDArray();
            using (var op = new Operator("_mod"))
            {
                op.Set(lhs, rhs).Invoke(ret);
            }

            return ret;
        }

        public static NDArray operator >(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastGreater(lhs, rhs);
        }

        public static NDArray operator >=(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastGreaterEqual(lhs, rhs);
        }

        public static NDArray operator >(NDArray lhs, float rhs)
        {
            return nd.GreaterScalar(lhs, rhs);
        }

        public static NDArray operator >=(NDArray lhs, float rhs)
        {
            return nd.GreaterEqualScalar(lhs, rhs);
        }

        public static NDArray operator >(float lhs, NDArray rhs)
        {
            return nd.GreaterScalar(rhs, lhs);
        }

        public static NDArray operator >=(float lhs, NDArray rhs)
        {
            return nd.GreaterEqualScalar(rhs, lhs);
        }

        public static NDArray operator <(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastLesser(lhs, rhs);
        }

        public static NDArray operator <=(NDArray lhs, NDArray rhs)
        {
            return nd.BroadcastLesserEqual(lhs, rhs);
        }

        public static NDArray operator <(NDArray lhs, float rhs)
        {
            return nd.LesserScalar(lhs, rhs);
        }

        public static NDArray operator <=(NDArray lhs, float rhs)
        {
            return nd.LesserEqualScalar(lhs, rhs);
        }

        public static NDArray operator <(float lhs, NDArray rhs)
        {
            return nd.LesserScalar(rhs, lhs);
        }

        public static NDArray operator <=(float lhs, NDArray rhs)
        {
            return nd.LesserEqualScalar(rhs, lhs);
        }

        public virtual NDArray Reshape(Shape shape, bool reverse = false)
        {
            var dims = shape.Data.Select(s => s);
            NativeMethods.MXNDArrayReshape64(GetHandle(), shape.Dimension, dims.ToArray(), reverse, out var handle);
            return new NDArray(handle);
        }

        public virtual NDArray Reshape(params int[] shape)
        {
            var targetShape = new int[shape.Length];
            long prod = -1 * shape.Aggregate(1L, (a, b) => a * b);
            for (var i = 0; i < targetShape.Length; i++)
                if (shape[i] != -1)
                    targetShape[i] = shape[i];
                else
                    targetShape[i] = Convert.ToInt32(Size / prod);

            return Reshape(new Shape(targetShape));

            //return Reshape(new Shape(shape));
        }

        public NDArray Ravel()
        {
            var n = Shape[0];
            var m = Size / n;
            return Reshape(new Shape(n, m));
        }

        public NDArray Squeeze(int? axis, bool inplace = false)
        {
            if (!inplace)
            {
                return nd.Squeeze(this, new Shape(axis));
            }
            else
            {
                var new_shape = this.Shape.Data;
                
                if (axis.HasValue)
                {
                    var axes = new List<int>() { axis.Value };
                    Debug.Assert(axes.Count == new HashSet<int>(axes).Count, "axis contains duplicate which is not allowed.");
                    var resolved_axes = (from i in axes
                                         select i >= 0 ? i : i + this.Shape.Dimension).ToList();

                    Enumerable.Zip(axes, resolved_axes, (arg_axis, actual_axis) => 
                    {
                        Debug.Assert((-new_shape.Count <= arg_axis) && (arg_axis < new_shape.Count), $"axis {arg_axis} is out of range for {new_shape.Count}d array");
                        var axis_size = new_shape[actual_axis];
                        Debug.Assert(axis_size == 1, $"Squeeze target axis {arg_axis} must be size 1, got {axis_size}.");
                        return axes;
                    });

                    foreach (var i in resolved_axes.OrderByDescending(_p_1 => _p_1).ToList())
                    {
                        new_shape.Remove(i);
                    }
                }
                else
                {
                    foreach (var i in Enumerable.Range(0, new_shape.Count).Reverse())
                    {
                        if (new_shape[i] == 1)
                        {
                            new_shape.Remove(i);
                        }
                    }
                }

                if (new_shape.Count == 0)
                {
                    new_shape.Add(1);
                }

                return this.Reshape(new Shape(new_shape));
            }
        }

        public static void CheckBooleanArrayDimension(Shape array_shape, int axis, Shape bool_shape)
        {
            foreach (var _tup_1 in bool_shape.Data.Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1)))
            {
                var i = _tup_1.Item1;
                var val = _tup_1.Item2;
                if (array_shape[axis + i] != val)
                {
                    throw new Exception($"boolean index did not match indexed array along axis {axis + i}; size is {array_shape[axis + i]} but corresponding boolean size is {val}");
                }
            }
        }

        #endregion

        #region Overrides

        public override string ToString()
        {
            return DataType.Name + ": " + Shape;
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            _Blob.Dispose();
        }

        public static implicit operator ndarray(NDArray x) => x.AsNumpy();

        public static implicit operator NDArray(ndarray x) => new NDArray(x.NativePtr);

        public static implicit operator NDArray(NDArrayOrSymbol x) => new NDArray(x.NdX.NativePtr);

        public static implicit operator NDArray(int x) => nd.Array(new float[] { x }).AsType(DType.Int32);

        public static implicit operator NDArray(long x) => nd.Array(new float[] { x }).AsType(DType.Int64);

        public static implicit operator NDArray(float x) => nd.Array(new float[] { x });

        public static implicit operator NDArray(double x) => nd.Array(new float[] { Convert.ToSingle(x) }).AsType(DType.Float64);

        public static implicit operator NDArray(OpenCvSharp.Mat x) => LoadCV2Mat(x);

        public static implicit operator OpenCvSharp.Mat(NDArray x)
        {
            var buffer = x.AsType(DType.UInt8).GetBuffer();
            var (h, w, c) = x.Shape;
            OpenCvSharp.Mat mat = new OpenCvSharp.Mat(new OpenCvSharp.Size(w, h), OpenCvSharp.MatType.CV_8UC3);
            unsafe
            {
                Buffer.MemoryCopy(buffer.GetMemPtr().ToPointer(), mat.Data.ToPointer(), buffer.Length, buffer.Length);
            }

            OpenCvSharp.Cv2.CvtColor(mat, mat, OpenCvSharp.ColorConversionCodes.RGB2BGR);
            return mat;
        }

        #endregion

        #endregion
    }
}