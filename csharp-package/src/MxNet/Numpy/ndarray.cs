using MxNet.Interop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NDArrayHandle = System.IntPtr;
using mx_uint = System.UInt32;
using mx_float = System.Single;
using size_t = System.UInt64;
using System.Runtime.InteropServices;
using System.Diagnostics;
using MxNet.ND.Numpy;
using System.IO.Compression;
using System.IO;

namespace MxNet.Numpy
{
    public partial class ndarray : DisposableMXNetObject
    {
        #region Fields
        private static readonly List<string>
           CastStorageStypeConvert = new List<string> { "default", "row_sparse", "csr" };
        internal NDBlob _Blob;

        public Context ctx
        {
            get
            {
                return GetContext();
            }
        }

        public Shape shape => new Shape(GetShape().ToArray());

        public DType dtype => DType.GetType(GetDType());

        public StorageStype stype => (StorageStype)StorageType();

        public ndarray T
        {
            get
            {
                return np.transpose(this);
            }
        }

        #endregion

        #region Constructors

        public ndarray()
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArrayCreateNone(out var @out), NativeMethods.OK);

            NativePtr = @out;
            _Blob = new NDBlob(@out);
        }

        internal ndarray(NDArrayHandle handle)
        {
            if (handle == NDArrayHandle.Zero)
                throw new ArgumentException("Can not pass IntPtr.Zero", nameof(handle));

            NativePtr = handle;
            _Blob = new NDBlob(handle);
        }

        public ndarray(Shape shape, bool delayAlloc = true, Context ctx = null, DType dtype = null)
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

        public ndarray(Array data, Shape shape, Context ctx = null, DType dtype = null)
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
            NativeMethods.MXNDArraySyncCopyFromCPU(@out, datagch.AddrOfPinnedObject(), (uint)shape.Size);
            datagch.Free();

            NativePtr = @out;
            _Blob = new NDBlob(@out);
        }

        public ndarray(Array data, Context ctx = null, DType dtype = null)
        {
            if (ctx == null)
                ctx = Context.CurrentContext;

            var shapeData = new List<int>();

            for (int i = 0; i < data.Rank; i++)
            {
                shapeData.Add(data.GetLength(i));
            }

            var shape = new Shape(shapeData.ToArray());

            if (dtype == null)
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

        public virtual long size
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

        public int ndim => shape.Dimension;

        public bool FreshGrad
        {
            get
            {
                NativeMethods.MXNDArrayGetGradState(NativePtr, out var freshGrad);
                return freshGrad;
            }
            set => NativeMethods.MXNDArraySetGradState(NativePtr, value);
        }

        public Array data => AsArray();

        public ndarray grad
        {
            get
            {
                NativeMethods.MXNDArrayGetGrad(NativePtr, out var h);
                return new ndarray(h);
            }
        }

        #endregion

        #region Methods

        public ndarray Copy()
        {
            var ret = new ndarray(shape);
            using (var op = new Operator("_copyto"))
            {
                op.Set(this).Invoke(ret);
            }

            return ret;
        }

        public ndarray CopyTo(ndarray other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            using (var op = new Operator("_copyto"))
            {
                op.Set(this).Invoke(other);
            }

            return other;
        }

        public ndarray ChangeContext(Context ctx)
        {
            var result = new ndarray(shape, true, ctx, dtype);
            CopyTo(result);
            return result;
        }

        public Context GetContext()
        {
            NativeMethods.MXNDArrayGetContext(NativePtr, out var out_dev_type, out var out_dev_id);
            return new Context((DeviceType)out_dev_type, out_dev_id);
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
                    arrayMap[name] = new ndarray(array[i]);
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

            Logging.CHECK_EQ(NativeMethods.MXNDArraySave(fileName, (uint)args.Length, args, keys), NativeMethods.OK);
        }

        public static void Save(string fileName, NDArrayList arrayList)
        {
            var args = arrayList.Select(array => array.GetHandle()).ToArray();
            Logging.CHECK_EQ(NativeMethods.MXNDArraySave(fileName, (uint)args.Length, args, null), NativeMethods.OK);
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
            Marshal.Copy(outArrPtr, outArr, 0, (int)outSize);


            if (outNameSize == 0)
            {
                for (var i = 0; i < outArr.Length; i++) data.Add(i.ToString(), new ndarray(outArr[i]));
            }
            else
            {
                var outNames = new IntPtr[outNameSize];
                Marshal.Copy(outNamesPtr, outNames, 0, (int)outNameSize);

                for (var i = 0; i < outArr.Length; i++)
                {
                    var key = Marshal.PtrToStringAnsi(outNames[i]);
                    if (!string.IsNullOrEmpty(key)) data.Add(key, new ndarray(outArr[i]));
                }
            }
        }

        public static NDArrayDict Load(string filename)
        {
            Load(filename, out var r);
            return r;
        }

        public static ndarray LoadFromBuffer(byte[] buffer)
        {
            NativeMethods.MXNDArrayLoadFromRawBytes(buffer, buffer.Length, out var handle);
            return new ndarray(handle);
        }

        public static ndarray LoadCV2Mat(OpenCvSharp.Mat img, Context context = null)
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

            var ret = new ndarray(bytes, s, context, dtype: DType.UInt8);
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

        public static ndarray NewFromSharedMem(int shared_pid, int shared_id, Shape shape, DType dtype)
        {
            NativeMethods.MXNDArrayCreateFromSharedMem(shared_pid, shared_id, shape.Data.ToArray(), shape.Dimension,
                dtype.Index, out var handle);
            return new ndarray(handle);
        }

        public (int, int, Shape, DType) ToSharedMem()
        {
            NativeMethods.MXNDArrayGetSharedMemHandle(NativePtr, out var shared_pid, out var shared_id);
            return (shared_pid, shared_id, shape, dtype);
        }

        public ndarray Cast(DType dtype)
        {
            return new Operator("cast")
                .SetParam("dtype", dtype)
                .SetInput("data", this)
                .Invoke();
        }

        public void Constant(float scalar)
        {
            var x = np.full(this.shape, scalar, this.dtype, ctx: this.ctx);
            this.NativePtr = x.NativePtr;
            x.Dispose();
        }

        public ndarray SliceAxis(int axis, int begin, int? end)
        {
            var @out = new ndarray();
            if (end.HasValue)
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

        public virtual ndarray Slice(int begin, int? end)
        {
            Logging.CHECK_EQ(NativeMethods.MXNDArraySlice(GetHandle(), begin, end.Value, out var handle),
                NativeMethods.OK);
            return new ndarray(handle);
        }

        public ndarray Slice(Shape begin, Shape end, Shape step = null)
        {
            if (step == null) step = new Shape();

            return new Operator("slice")
                .SetParam("begin", begin)
                .SetParam("end", end)
                .SetParam("step", step)
                .SetInput("data", this)
                .Invoke();
        }

        public ndarray SliceAssignScalar(double value, Shape begin, Shape end, Shape step)
        {
            return nd.SliceAssignScalar(this, scalar: value, begin: begin, end: end, step: step);
        }

        public ndarray SliceAssign(ndarray rhs, Shape begin, Shape end, Shape step)
        {
            return nd.SliceAssign(this, rhs: rhs, begin: begin, end: end, step: step);
        }

        public void SyncCopyFromCPU(Array data, ulong size)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var resize = size > 0;
            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);

            NativeMethods.MXNDArraySyncCopyFromCPU(NativePtr, datagch.AddrOfPinnedObject(), (uint)size);

            datagch.Free();
        }

        public virtual void SyncCopyFromCPU(Array data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            var datagch = GCHandle.Alloc(data, GCHandleType.Pinned);
            NativeMethods.MXNDArraySyncCopyFromCPU(NativePtr, datagch.AddrOfPinnedObject(), (uint)data.Length);
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
            NativeMethods.MXNDArraySyncCopyToCPU(NativePtr, datagch.AddrOfPinnedObject(), (ulong)size);

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
            var size = this.size;
            Array data = null;

            switch (dtype.Name)
            {
                case "bool":
                    data = Array.CreateInstance(typeof(bool), size);
                    break;
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
            NativeMethods.MXNDArraySyncCopyToCPU(NativePtr, datagch.AddrOfPinnedObject(), (ulong)size);
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

        public virtual ndarray AsType(DType dtype)
        {
            return nd.Cast(this, dtype);
        }

        public ndarray AsInContext(Context context)
        {
            if (this.ctx.ToString() == context.ToString())
                return this;

            return ChangeContext(context);
        }

        private int StorageType()
        {
            NativeMethods.MXNDArrayGetStorageType(GetHandle(), out var out_storage_type);
            return out_storage_type;
        }

        public virtual NumpyDotNet.ndarray AsNumpy()
        {
            NumpyDotNet.ndarray x = NumpyDotNet.np.array(AsArray()); ;

            var npShape = new List<int>();
            foreach (var item in shape.Data)
            {
                if (item == 0)
                    continue;

                npShape.Add(item);
            }

            return NumpyDotNet.np.reshape(x, new NumpyDotNet.shape(npShape.ToArray()));
        }

        public ndarray this[int index]
        {
            get
            {
                var x = Slice(index, index + 1);
                var new_shape = x.shape.Data.ToList();
                new_shape.RemoveAt(0);
                return x.reshape(new Shape(new_shape));
            }
        }

        public ndarray this[int begin, int end]
        {
            get
            {
                return Slice(begin, end);
            }
        }

        public ndarray this[ndarray begin, ndarray end]
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public ndarray this[string slice]
        {
            get
            {
                if (string.IsNullOrEmpty(slice))
                    return this;

                var (begin, end) = MxUtil.GetSliceNotation(slice, shape);

                return Slice(begin, end);
            }
            set
            {
                if (string.IsNullOrEmpty(slice))
                    value.CopyTo(this);

                var (begin, end) = MxUtil.GetSliceNotation(slice, shape);
                ndarray output = null;

                if (value.size == 1)
                    output = nd.SliceAssignScalar(this, begin, end, value.AsScalar<double>());
                else
                    output = nd.SliceAssign(this, value, begin, end);

                output.CopyTo(this);
            }
        }

        public ndarray this[ndarray indices] => nd.Take(this, indices);

        public void AttachGrad(OpGradReq grad_req = OpGradReq.Write, StorageStype? stype = null)
        {
            ndarray grad = np.zeros_like(this);
            if (stype.HasValue)
                grad = grad.ToSType(stype.Value);

            Autograd.MarkVariables(this, grad, grad_req);
        }

        public ndarray Detach()
        {
            NativeMethods.MXNDArrayDetach(GetHandle(), out var hdl);
            return new ndarray(hdl);
        }

        public void Backward(ndarray out_grad = null, bool retain_graph = false, bool train_mode = true)
        {
            var ograd_handles = new List<NDArrayHandle>();
            var var_handles = new List<NDArrayHandle>();
            //var grad_handles = new List<NDArrayHandle>();
            if (out_grad != null)
                ograd_handles.Add(out_grad.GetHandle());
            else
                ograd_handles.Add(new NDArrayHandle());

            NativeMethods.MXAutogradBackwardEx(1, new NDArrayHandle[1] { NativePtr }, ograd_handles.ToArray(),
                0, var_handles.ToArray(), retain_graph ? 1 : 0,
                0, train_mode ? 1 : 0, out var grad_handles, out var grad_count);
        }

        public ndarray ToSType(StorageStype stype)
        {
            if (stype == StorageStype.Csr && this.shape.Dimension != 2)
            {
                throw new System.Exception("To convert to a CSR, the NDArray should be 2 Dimensional. Current shape is " + this.shape);
            }

            if (this.stype == stype)
                return this;

            return new Operator("cast_storage")
                .SetParam("stype", MxUtil.EnumToString<StorageStype>(stype, CastStorageStypeConvert))
                .SetInput("data", this)
                .Invoke();
        }

        #region Operators

        public static ndarray operator +(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastAdd(lhs, rhs);
        }

        public static ndarray operator +(ndarray lhs, float scalar)
        {
            return nd.PlusScalar(lhs, scalar);
        }

        public static ndarray operator +(float scalar, ndarray rhs)
        {
            return nd.PlusScalar(rhs, scalar);
        }

        public static ndarray operator -(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastSub(lhs, rhs);
        }

        public static ndarray operator -(ndarray lhs, float scalar)
        {
            return nd.MinusScalar(lhs, scalar);
        }

        public static ndarray operator -(float scalar, ndarray rhs)
        {
            return nd.RminusScalar(rhs, scalar);
        }

        public static ndarray operator *(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastMul(lhs, rhs);
        }

        public static ndarray operator *(ndarray lhs, float scalar)
        {
            return nd.MulScalar(lhs, scalar);
        }

        public static ndarray operator *(float scalar, ndarray rhs)
        {
            return nd.MulScalar(rhs, scalar);
        }

        public static ndarray operator /(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastDiv(lhs, rhs);
        }

        public static ndarray operator /(ndarray lhs, float scalar)
        {
            return nd.DivScalar(lhs, scalar);
        }

        public static ndarray operator /(float scalar, ndarray rhs)
        {
            return nd.RdivScalar(rhs, scalar);
        }

        public static ndarray operator %(ndarray lhs, float scalar)
        {
            var ret = new ndarray();
            using (var op = new Operator("_mod_scalar"))
            {
                op.Set(lhs, scalar).Invoke(ret);
            }

            return ret;
        }

        public static ndarray operator %(ndarray lhs, ndarray rhs)
        {
            var ret = new ndarray();
            using (var op = new Operator("_mod"))
            {
                op.Set(lhs, rhs).Invoke(ret);
            }

            return ret;
        }

        public static ndarray operator >(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastGreater(lhs, rhs);
        }

        public static ndarray operator >=(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastGreaterEqual(lhs, rhs);
        }

        public static ndarray operator >(ndarray lhs, float rhs)
        {
            return nd.GreaterScalar(lhs, rhs);
        }

        public static ndarray operator >=(ndarray lhs, float rhs)
        {
            return nd.GreaterEqualScalar(lhs, rhs);
        }

        public static ndarray operator >(float lhs, ndarray rhs)
        {
            return nd.GreaterScalar(rhs, lhs);
        }

        public static ndarray operator >=(float lhs, ndarray rhs)
        {
            return nd.GreaterEqualScalar(rhs, lhs);
        }

        public static ndarray operator <(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastLesser(lhs, rhs);
        }

        public static ndarray operator <=(ndarray lhs, ndarray rhs)
        {
            return nd.BroadcastLesserEqual(lhs, rhs);
        }

        public static ndarray operator <(ndarray lhs, float rhs)
        {
            return nd.LesserScalar(lhs, rhs);
        }

        public static ndarray operator <=(ndarray lhs, float rhs)
        {
            return nd.LesserEqualScalar(lhs, rhs);
        }

        public static ndarray operator <(float lhs, ndarray rhs)
        {
            return nd.LesserScalar(rhs, lhs);
        }

        public static ndarray operator <=(float lhs, ndarray rhs)
        {
            return nd.LesserEqualScalar(rhs, lhs);
        }

        public static ndarray operator -(ndarray x)
        {
            return np.negative(x);
        }

        public virtual ndarray reshape(Shape shape, bool reverse = false)
        {
            return nd_np_ops.reshape(this, shape, reverse);
        }

        public virtual ndarray reshape(params int[] shape)
        {
            var targetShape = new int[shape.Length];
            long prod = -1 * shape.Aggregate(1L, (a, b) => a * b);
            for (var i = 0; i < targetShape.Length; i++)
                if (shape[i] != -1)
                    targetShape[i] = shape[i];
                else
                    targetShape[i] = Convert.ToInt32(size / prod);

            return reshape(new Shape(targetShape));
        }

        public ndarray Ravel()
        {
            var n = shape[0];
            var m = size / n;
            return reshape(new Shape(n, m));
        }

        public ndarray Squeeze(int? axis, bool inplace = false)
        {
            if (!inplace)
            {
                return nd.Squeeze(this, new Shape(axis));
            }
            else
            {
                var new_shape = this.shape.Data;

                if (axis.HasValue)
                {
                    var axes = new List<int>() { axis.Value };
                    Debug.Assert(axes.Count == new HashSet<int>(axes).Count, "axis contains duplicate which is not allowed.");
                    var resolved_axes = (from i in axes
                                         select i >= 0 ? i : i + this.shape.Dimension).ToList();

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

                return this.reshape(new Shape(new_shape));
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
            return dtype.Name + ": " + shape;
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            _Blob.Dispose();
        }

        public static implicit operator ndarray(int x) => np.array(new float[] { x }).AsType(DType.Int32);

        public static implicit operator ndarray(long x) => np.array(new float[] { x }).AsType(DType.Int64);

        public static implicit operator ndarray(float x) => np.array(new float[] { x });

        public static implicit operator ndarray(double x) => np.array(new float[] { Convert.ToSingle(x) }).AsType(DType.Float64);

        public static implicit operator ndarray(OpenCvSharp.Mat x) => LoadCV2Mat(x);

        public static implicit operator OpenCvSharp.Mat(ndarray x)
        {
            var buffer = x.AsType(DType.UInt8).GetBuffer();
            var (h, w, c) = x.shape;
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
