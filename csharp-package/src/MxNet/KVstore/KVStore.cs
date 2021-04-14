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
using System.Diagnostics;
using System.IO;
using System.Linq;
using MxNet.Interop;
using MxNet.Optimizers;
using Newtonsoft.Json;
using KVStoreHandle = System.IntPtr;

namespace MxNet.KVstore
{
    public enum KVStoreCommandType
    {
        kController = 0,
        kSetMultiPrecision,
        kStopServer,
        kSyncMode,
        kSetGradientCompression,
        kSetProfilerParams
    }

    public class KVStore : KVStoreBase, IDisposable
    {
        internal Updater _updater;

        internal KVStoreHandle handle;
        internal IntPtr str_updater_func;
        internal IntPtr updater_func;
        internal bool _is_p3;

        public static int GetKVStoreServerCommandType(string command)
        {
            var command_types = new Dictionary<string, int> {
                {
                    "kController",
                    0},
                {
                    "kSetMultiPrecision",
                    1},
                {
                    "kStopServer",
                    2},
                {
                    "kSyncMode",
                    3},
                {
                    "kSetGradientCompression",
                    4},
                {
                    "kSetProfilerParams",
                    5}};
            Debug.Assert(command_types.ContainsKey(command), "Unknown command type to send to server");
            return command_types[command];
        }

        public KVStore(KVStoreHandle handle)
        {
            this.handle = handle;
            _updater = null;
            updater_func = IntPtr.Zero;
            str_updater_func = IntPtr.Zero;
            _is_p3 = Environment.GetEnvironmentVariable("DMLC_PS_VAN_TYPE") == "p3";
        }

        public override int Rank
        {
            get
            {
                NativeMethods.MXKVStoreGetRank(handle, out var ret);
                return ret;
            }
        }

        public override string Type
        {
            get
            {
                var name = "";
                NativeMethods.MXKVStoreGetType(handle, name);
                return name;
            }
        }

        public override int NumWorkers
        {
            get
            {
                NativeMethods.MXKVStoreGetGroupSize(handle, out var ret);
                return ret;
            }
        }

        public void Dispose()
        {
            NativeMethods.MXKVStoreFree(handle);
        }

        public override void Broadcast(string key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            List<string> cv_keys = new List<string>();
            for(int i = 0; i< value.Length; i++)
            {
                cv_keys.Add(key);
            }

            List<string> co_keys = new List<string>();
            for (int i = 0; i < @out.Length; i++)
            {
                co_keys.Add(key);
            }

            NativeMethods.MXKVStoreBroadcastEx(handle, cv_keys.Count, cv_keys.ToArray(), co_keys.Count, co_keys.ToArray(), value.Handles, @out.Handles, priority);
            Pull(key, @out, priority);
        }

        public override void Broadcast(int key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            List<int> cv_keys = new List<int>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            List<int> co_keys = new List<int>();
            for (int i = 0; i < @out.Length; i++)
            {
                co_keys.Add(key);
            }

            NativeMethods.MXKVStoreBroadcast(handle, cv_keys.Count, cv_keys.ToArray(), co_keys.Count, co_keys.ToArray(), value.Handles, @out.Handles, priority);
        }

        public void Init(string key, NDArrayList value)
        {
            List<string> cv_keys = new List<string>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }
            
            NativeMethods.MXKVStoreInitEx(handle, cv_keys.Count, cv_keys.ToArray(), value.Handles);
        }

        public void Init(int key, NDArrayList value)
        {
            List<int> cv_keys = new List<int>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            NativeMethods.MXKVStoreInit(handle, cv_keys.Count, cv_keys.ToArray(), value.Handles);
        }

        public void Push(string key, NDArrayList value, int priority = 0)
        {
            List<string> cv_keys = new List<string>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            NativeMethods.MXKVStorePushEx(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(value), priority);
        }

        public void Push(int key, NDArrayList value, int priority = 0)
        {
            List<int> cv_keys = new List<int>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            NativeMethods.MXKVStorePush(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(value), priority);
        }

        public void Pull(string key, NDArrayList @out = null, int priority = 0, bool ignore_sparse = true)
        {
            List<string> cv_keys = new List<string>();
            for (int i = 0; i < @out.Length; i++)
            {
                cv_keys.Add(key);
            }

            NativeMethods.MXKVStorePullWithSparseEx(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(@out), priority,
                ignore_sparse);
        }

        public void Pull(int key, NDArrayList @out = null, int priority = 0, bool ignore_sparse = true)
        {
            List<int> cv_keys = new List<int>();
            for (int i = 0; i < @out.Length; i++)
            {
                cv_keys.Add(key);
            }

            NativeMethods.MXKVStorePullWithSparse(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(@out), priority,
                ignore_sparse);
        }

        public override void PushPull(string key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            List<string> cv_keys = new List<string>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            List<string> co_keys = new List<string>();
            for (int i = 0; i < @out.Length; i++)
            {
                co_keys.Add(key);
            }

            NativeMethods.MXKVStorePushPullEx(handle, cv_keys.Count, cv_keys.ToArray(), co_keys.Count, co_keys.ToArray(), value.Handles,
                MxUtil.GetNDArrayHandles(@out), priority);
        }

        public override void PushPull(int key, NDArrayList value, NDArrayList @out, int priority = 0)
        {
            List<int> cv_keys = new List<int>();
            for (int i = 0; i < value.Length; i++)
            {
                cv_keys.Add(key);
            }

            List<int> co_keys = new List<int>();
            for (int i = 0; i < @out.Length; i++)
            {
                co_keys.Add(key);
            }

            NativeMethods.MXKVStorePushPull(handle, cv_keys.Count, cv_keys.ToArray(), co_keys.Count, co_keys.ToArray(), value.Handles,
                MxUtil.GetNDArrayHandles(@out), priority);
        }

        public void RowSparsePull(string key, NDArrayList @out, int priority = 0, NDArrayList row_ids = null)
        {
            if (@out == null)
                throw new ArgumentNullException("@out");

            if (row_ids == null)
                throw new ArgumentNullException("row_ids");

            List<string> cv_keys = new List<string>();
            for (int i = 0; i < @out.Length; i++)
            {
                cv_keys.Add(key);
            }

            var first_out = new NDArrayList();
            if (row_ids.Length == 1)
                first_out.Add(@out[0]);
            else
                first_out = @out.ToList();

            NativeMethods.MXKVStorePullRowSparseEx(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(first_out.ToArray()), MxUtil.GetNDArrayHandles(row_ids), priority);
        }

        public void RowSparsePull(int key, NDArrayList @out, int priority = 0, NDArrayList row_ids = null)
        {
            if (@out == null)
                throw new ArgumentNullException("@out");

            if (row_ids == null)
                throw new ArgumentNullException("row_ids");

            List<int> cv_keys = new List<int>();
            for (int i = 0; i < @out.Length; i++)
            {
                cv_keys.Add(key);
            }

            var first_out = new NDArrayList();
            if (row_ids.Length == 1)
                first_out.Add(@out[0]);
            else
                first_out = @out.ToList();

            NativeMethods.MXKVStorePullRowSparse(handle, cv_keys.Count, cv_keys.ToArray(), MxUtil.GetNDArrayHandles(first_out.ToArray()), MxUtil.GetNDArrayHandles(row_ids), priority);
        }


        public override bool IsCapable(string capability)
        {
            if (OPTIMIZER == capability.ToLower())
                return true;
            throw new MXNetException("Unknown capability: " + capability);
        }

        public void SetGradientCompression(Dictionary<string, object> compression_params)
        {
            if (Type.Contains("device") || Type.Contains("dist"))
            {
                var ckeys = compression_params.Keys.ToArray();
                var cvals = compression_params.Values.Select(x => x.ToString()).ToArray();

                NativeMethods.MXKVStoreSetGradientCompression(handle, compression_params.Count, ckeys, cvals);
            }
            else
            {
                throw new Exception("Gradient compression is not supported for this type of kvstore");
            }
        }

        public override void LoadOptimizerStates(string fname)
        {
            var data = File.ReadAllText(fname);
            this._updater.SetStates(data);
        }

        public override void SaveOptimizerStates(string fname, bool dump_optimizer = false)
        {
            Debug.Assert(this._updater != null, "Cannot save states for distributed training");
            File.WriteAllText(fname, this._updater.GetStates(dump_optimizer));
        }

        public override void SetOptimizer(Optimizer optimizer)
        {
            NativeMethods.MXKVStoreIsWorkerNode(out var is_worker);
            if (Type.Contains("dist") && is_worker > 0)
            {
                var optim_str = JsonConvert.SerializeObject(optimizer);
                var cmd = (int) KVStoreCommandType.kController;
                NativeMethods.MXKVStoreSendCommmandToServers(handle, cmd, optim_str);
                if (optimizer.MultiPrecision)
                    NativeMethods.MXKVStoreSendCommmandToServers(handle, (int) KVStoreCommandType.kSetMultiPrecision,
                        "");
            }
            else
            {
                SetUpdater(Optimizer.GetUpdater(optimizer));
            }
        }

        public virtual void Barrier()
        {
            NativeMethods.MXKVStoreBarrier(this.handle);
        }

        public virtual void SendCommandToServers(int head, string body)
        {
            NativeMethods.MXKVStoreSendCommmandToServers(this.handle, head, body);
        }

        private UpdaterHandle UpdaterWrapper(Updater updater)
        {
            UpdaterHandle func = (key, lhs_handle, rhs_handle, _) =>
            {
                updater.Call(key, new NDArray(lhs_handle), new NDArray(rhs_handle));
            };

            return func;
        }

        private UpdaterHandleStr UpdaterWrapperStr(Updater updater)
        {
            UpdaterHandleStr func = (key, lhs_handle, rhs_handle, _) =>
            {
                updater.Call(Convert.ToInt32(key), new NDArray(lhs_handle), new NDArray(rhs_handle));
            };

            return func;
        }

        public void SetUpdater(Updater updater)
        {
            _updater = updater;
            updater_func = UpdaterWrapper(updater).Method.MethodHandle.GetFunctionPointer();
            str_updater_func = UpdaterWrapperStr(updater).Method.MethodHandle.GetFunctionPointer();
            NativeMethods.MXKVStoreSetUpdaterEx(handle, updater_func, str_updater_func, IntPtr.Zero);
        }

        internal delegate void UpdaterHandle(int key, IntPtr lhs_handle, IntPtr rhs_handle, IntPtr _);

        internal delegate void UpdaterHandleStr(string key, IntPtr lhs_handle, IntPtr rhs_handle, IntPtr _);
    }
}