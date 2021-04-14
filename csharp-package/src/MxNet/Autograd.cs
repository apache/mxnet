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
using System.Linq;
using MxNet.Interop;
using MxNet.Numpy;
using MxNet.Sym.Numpy;

namespace MxNet
{
    public class Autograd
    {
        public static bool SetRecording(bool is_recording)
        {
            var prev = 0;
            NativeMethods.MXAutogradSetIsRecording(Convert.ToInt32(is_recording), ref prev);

            return Convert.ToBoolean(prev);
        }

        public static bool SetTraining(bool train_mode)
        {
            var prev = 0;
            NativeMethods.MXAutogradSetIsTraining(Convert.ToInt32(train_mode), ref prev);

            return Convert.ToBoolean(prev);
        }

        public static bool IsRecording()
        {
            var curr = 0;
            NativeMethods.MXAutogradIsRecording(ref curr);

            return Convert.ToBoolean(curr);
        }

        public static bool IsTraining()
        {
            var curr = 0;
            NativeMethods.MXAutogradIsTraining(ref curr);

            return Convert.ToBoolean(curr);
        }

        public static _RecordingStateScope Record(bool train_mode = true)
        {
            return new _RecordingStateScope(true, train_mode);
        }

        public static _RecordingStateScope Pause(bool train_mode = true)
        {
            return new _RecordingStateScope(false, train_mode);
        }

        public static _RecordingStateScope TrainMode()
        {
            return new _RecordingStateScope(null, true);
        }

        public static _RecordingStateScope PredictMode()
        {
            return new _RecordingStateScope(null, false);
        }

        public static void MarkVariables(NDArrayList variables, NDArrayList gradients,
            OpGradReq grad_reqs = OpGradReq.Write)
        {
            var gradReqs = new int[variables.Length];
            for (var i = 0; i < gradReqs.Length; i++) gradReqs[i] = (int) OpGradReq.Write;

            NativeMethods.MXAutogradMarkVariables(variables.Length, MxUtil.GetNDArrayHandles(variables), gradReqs,
                MxUtil.GetNDArrayHandles(gradients));
        }

        private static (IntPtr[], IntPtr[]) ParseHead(NDArrayList heads, NDArrayList head_grads)
        {
            IntPtr[] headHandles = null;
            IntPtr[] headGradHandles = null;

            headHandles = MxUtil.GetNDArrayHandles(heads);

            if (head_grads == null)
            {
                headGradHandles = new IntPtr[heads.Length];
                for (var i = 0; i < headHandles.Length; i++) headGradHandles[i] = IntPtr.Zero;
            }
            else
            {
                if (heads.Length != head_grads.Length)
                    throw new ArgumentException("heads and head_grads must be lists of the same length");

                headGradHandles = MxUtil.GetNDArrayHandles(head_grads);
            }

            return (headHandles, headGradHandles);
        }

        public static void Backward(NDArrayList heads, NDArrayList head_grads = null, bool retain_graph = false,
            bool train_mode = true)
        {
            var (head_handles, head_grads_handles) = ParseHead(heads, head_grads);

            NativeMethods.MXAutogradBackwardEx(head_handles.Length, head_handles, head_grads_handles, 0,
                new IntPtr[] { }, Convert.ToInt32(retain_graph),
                0, Convert.ToInt32(train_mode), out var grad_handles, out var grad_count);
        }

        public static NDArrayList Grad(NDArrayList heads, NDArrayList variables, NDArrayList head_grads = null,
            bool retain_graph = false, bool create_graph = true, bool train_mode = true)
        {
            var (head_handles, head_grads_handles) = ParseHead(heads, head_grads);

            //var grad_handles = new IntPtr[head_handles.Length];
            //var grad_stypes = new int[head_handles.Length];

            NativeMethods.MXAutogradBackwardEx(head_handles.Length, head_handles, head_grads_handles, variables.Length,
                MxUtil.GetNDArrayHandles(variables), Convert.ToInt32(retain_graph),
                Convert.ToInt32(create_graph), Convert.ToInt32(train_mode), out var grad_handles, out var grad_stypes);

            var result = new NDArrayList();
            foreach (var item in grad_handles) result.Add(new NDArray(item));

            return result.ToArray();
        }

        internal static _Symbol GetSymbol(ndarray x)
        {
            var hdl = IntPtr.Zero;
            NativeMethods.MXAutogradGetSymbol(x.GetHandle(), hdl);
            return new _Symbol(hdl);
        }

        public class _RecordingStateScope : MxDisposable
        {
            private bool? _enter_is_record;
            private bool? _enter_train_mode;
            private bool? _prev_is_record;
            private bool? _prev_train_mode;

            public _RecordingStateScope(bool? is_record, bool train_mode)
            {
                _enter_is_record = is_record;
                _enter_train_mode = train_mode;
                _prev_is_record = null;
                _prev_train_mode = null;
                With();
            }

            public override MxDisposable With()
            {
                if (_enter_is_record.HasValue)
                    _prev_is_record = SetRecording(_enter_is_record.Value);

                if (_enter_train_mode.HasValue)
                    _prev_train_mode = SetTraining(_enter_train_mode.Value);
                return this;
            }

            public override void Exit()
            {
                if (_enter_is_record.HasValue && _prev_is_record != _enter_is_record)
                    SetRecording(_prev_is_record.Value);

                if (_enter_train_mode.HasValue && _prev_train_mode != _enter_train_mode)
                    SetTraining(_prev_train_mode.Value);
            }
        }
    }
}