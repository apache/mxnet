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
using MxNet.Interop;
using mx_float = System.Single;
using DataIterHandle = System.IntPtr;
using System.Linq;

// ReSharper disable once CheckNamespace
namespace MxNet.IO
{
    public sealed class MXDataIter : DataIter
    {
        private bool _debug_at_begin = true;
        private bool _debug_skip_load;
        private readonly NDArray data;
        private readonly string data_name;
        private readonly MXDataIterMap DataIterMap = new MXDataIterMap();
        private DataBatch first_batch;
        private readonly DataIterHandle handle;
        private readonly NDArray label;
        private readonly string label_name;
        private int? next_res;
        private readonly Operator op;

        public MXDataIter(string mxdataiterType, string data_name = "data", string label_name = "softmax_label")
        {
            handle = DataIterMap.GetMXDataIterCreator(mxdataiterType);
            _debug_skip_load = false;
            first_batch = Next();
            data = first_batch.Data[0];
            label = first_batch.Label[0];
            this.data_name = data_name;
            this.label_name = label_name;
            BatchSize = data.Shape[0];
            op = new Operator(handle);
        }

        public MXDataIter(DataIterHandle handle, string data_name = "data", string label_name = "softmax_label")
        {
            this.handle = handle;
            _debug_skip_load = false;
            first_batch = Next();
            data = first_batch.Data[0];
            label = first_batch.Label[0];

            BatchSize = data.Shape[0];
            op = new Operator(handle);
        }

        public override DataDesc[] ProvideData
        {
            get { return new[] {new DataDesc(data_name, data.Shape, data.DataType)}; }
        }

        public override DataDesc[] ProvideLabel
        {
            get { return new[] {new DataDesc(label_name, label.Shape, label.DataType)}; }
        }

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            if (handle != DataIterHandle.Zero)
                NativeMethods.MXDataIterFree(handle);
        }

        public void DebugSkipLoad()
        {
            _debug_skip_load = true;
            Console.WriteLine("Set debug_skip_load to be true, will simply return first batch");
        }

        public override NDArrayList GetData()
        {
            NativeMethods.MXDataIterGetData(handle, out var hdl);
            return new NDArray(hdl);
        }

        public override int[] GetIndex()
        {
            var r = NativeMethods.MXDataIterGetIndex(handle, out var outIndex, out var outSize);
            Logging.CHECK_EQ(r, 0);

            var outIndexArray = InteropHelper.ToUInt64Array(outIndex, (uint) outSize);
            var ret = new int[outSize];
            for (var i = 0ul; i < outSize; ++i)
                ret[i] = (int) outIndexArray[i];

            return ret;
        }

        public override NDArrayList GetLabel()
        {
            NativeMethods.MXDataIterGetLabel(handle, out var hdl);
            return new NDArray(hdl);
        }

        public override int GetPad()
        {
            var r = NativeMethods.MXDataIterGetPadNum(handle, out var @out);
            return @out;
        }

        public override bool IterNext()
        {
            if (first_batch != null)
                return true;

            next_res = 0;
            NativeMethods.MXDataIterNext(handle, out next_res);
            return Convert.ToBoolean(next_res.Value);
        }

        public override bool End()
        {
            throw new NotImplementedException();
        }

        public override void Reset()
        {
            _debug_at_begin = true;
            first_batch = null;
            NativeMethods.MXDataIterBeforeFirst(handle);
        }

        public override DataBatch Next()
        {
            if (_debug_skip_load && !_debug_at_begin) return new DataBatch(GetData(), GetLabel(), GetPad(), GetIndex());

            if (first_batch != null)
            {
                var batch = first_batch;
                first_batch = null;
                return batch;
            }

            _debug_at_begin = false;
            next_res = 0;
            NativeMethods.MXDataIterNext(handle, out next_res);
            if (next_res.HasValue)
                return new DataBatch(GetData(), GetLabel(), GetPad(), GetIndex());
            throw new MXNetException("Stop Iteration");
        }

        public void SetParam(string key, string value)
        {
            op.SetParam(key, value);
        }

        public void SetParam(string key, NDArray value)
        {
            op.SetParam(key, value);
        }

        public void SetParam(string key, Symbol value)
        {
            op.SetParam(key, value);
        }

        public void SetParam(string key, object value)
        {
            op.SetParam(key, value);
        }

        public NDArrayList GetItems()
        {
            var output_vars = new NDArrayList();
            NativeMethods.MXDataIterGetItems(this.handle, out var num_output, output_vars.Handles);
            return output_vars;
        }

        public int Length
        {
            get
            {
                NativeMethods.MXDataIterGetLenHint(this.handle, out var length);
                if (length < 0)
                {
                    return 0;
                }

                return length;
            }
        }
    }
}