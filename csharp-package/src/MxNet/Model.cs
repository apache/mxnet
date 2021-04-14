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
using System.Linq;
using MxNet.Callbacks;
using MxNet.IO;
using MxNet.KVstore;
using MxNet.Gluon.Metrics;
using MxNet.Optimizers;

namespace MxNet
{
    public class MxModel
    {
        internal static (KVStore, bool) CreateSparseKVStore(KVStore kvstore)
        {
            Debug.Assert(kvstore.IsCapable(KVStore.OPTIMIZER), "KVStore with sparse weight requires optimizer support. However, type(kv) does not support optimizer. Please consider other kvstore backends (e.g. dist_device) instead.");
            return (kvstore, true);
        }

        internal static (KVStore, bool) CreateSparseKVStore(string kvstore)
        {
            return CreateSparseKVStore(KVStoreBase.Create(kvstore));
        }

        internal static (KVStore, bool) CreateKVStore(KVStore kvstore, int num_device, NDArrayDict arg_params)
        {
            var update_on_kvstore = true;
            if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MXNET_UPDATE_ON_KVSTORE")))
                update_on_kvstore = Convert.ToBoolean(Environment.GetEnvironmentVariable("MXNET_UPDATE_ON_KVSTORE"));

            if (kvstore == null)
                update_on_kvstore = false;
            else
                update_on_kvstore = !kvstore.IsCapable(KVStoreBase.OPTIMIZER);

            return (kvstore, update_on_kvstore);
        }

        internal static (KVStore, bool) CreateKVStore(string kvstore, int num_device, NDArrayDict arg_params)
        {
            KVStore kV = null;
            var update_on_kvstore = true;
            if (num_device == 1 && !kvstore.Contains("dist"))
            {
                kV = null;
            }
            else
            {
                kV = KVStoreBase.Create(kvstore);
                if (kvstore == "local")
                {
                    var max_size = arg_params.Values.Select(x => x.shape.Size).ToList().Max();
                    if (max_size > 1024 * 1024 * 16)
                        update_on_kvstore = false;
                }
            }

            if (kV == null)
                update_on_kvstore = false;
            else
                update_on_kvstore = !kV.IsCapable(KVStoreBase.OPTIMIZER);

            return (kV, update_on_kvstore);
        }

        internal static void InitializeKVStore(KVStore kvstore, List<NDArrayList> param_arrays, NDArrayDict arg_params,
            string[] param_names, bool update_on_kvstore)
        {
            for (int i = 0; i < param_arrays.Count; i++)
            {
                if (param_arrays[i].Length == 0)
                    continue;

                if (param_arrays[i][0] == null)
                    continue;

                var name = param_names[i];
                var param_on_devs = param_arrays[i];
                if (!update_on_kvstore || arg_params[name].stype != StorageStype.Default)
                {
                    kvstore.Init(name, arg_params[name]);
                }
                else
                {
                    kvstore.Broadcast(name, arg_params[name], @out: param_on_devs);
                }
            }
        }

        internal static void UpdateParamsOnKVStoreNCCL(List<NDArrayList> param_arrays, List<NDArrayList> grad_arrays,
            KVStore kvstore, string[] param_names)
        {
            List<int> valid_indices = new List<int>();
            int i = 0;
            grad_arrays.ForEach((x) => {
                valid_indices.Add(i);
                i++;
            });

            var valid_grad_arrays = valid_indices.Select(x => (grad_arrays[x])).ToArray();
            var valid_param_arrays = valid_indices.Select(x => (param_arrays[x])).ToArray();
            var valid_param_names = valid_indices.Select(x => (param_names[x])).ToArray();

            int size = valid_grad_arrays.Length;
            int start = 0;
            int batch = 16;
            if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MXNET_UPDATE_AGGREGATION_SIZE")))
                batch = Convert.ToInt32(Environment.GetEnvironmentVariable("MXNET_UPDATE_AGGREGATION_SIZE"));

            while(start < size)
            {
                int end = start + batch < size ? start + batch : size;
                var name_batch_list = valid_param_names.Skip(start).Take(end - start).ToArray();
                var grad_batch_list = valid_grad_arrays.Skip(start).Take(end - start).ToArray();
                var param_batch_list = valid_grad_arrays.Skip(start).Take(end - start).ToArray();

                for (int kvi = 0; kvi < name_batch_list.Length; kvi++)
                {
                    kvstore.Push(valid_param_names[kvi], valid_grad_arrays[kvi], -start);
                    kvstore.Pull(valid_param_names[kvi], param_batch_list[kvi], -start);
                }

                start = end;
            }
        }

        internal static void UpdateParamsOnKVStore(List<NDArrayList> param_arrays, List<NDArrayList> grad_arrays,
            KVStore kvstore, string[] param_names)
        {
            for (int index = 0; index < param_arrays.Count; index++)
            {
                var arg_list = param_arrays[index];
                var grad_list = grad_arrays[index];

                if (grad_list.Length == 0)
                    continue;

                if (grad_list[0] == null)
                    continue;

                string name = param_names[index];
                if (grad_list[0].stype == StorageStype.Default && arg_list[0].stype == StorageStype.Default)
                {
                    kvstore.PushPull(name, grad_list, @out: arg_list, priority: -index);
                }
                else
                {
                    kvstore.Push(name, grad_list, -index);
                    kvstore.Pull(name, arg_list, -index);
                }
            }
        }

        internal static void UpdateParams(List<NDArrayList> param_arrays, List<NDArrayList> grad_arrays,
            Updater updater, int num_device, KVStore kvstore, string[] param_names)
        {
            Dictionary<int, List<(int, NDArray, NDArray)>> updates = new Dictionary<int, List<(int, NDArray, NDArray)>>();
            for (int i = 0; i < num_device; i++)
            {
                updates.Add(i, new List<(int, NDArray, NDArray)>());
            }

            for (int i = 0; i < param_arrays.Count; i++)
            {
                var arg_list = param_arrays[i];
                var grad_list = grad_arrays[i];

                if (grad_list.Length == 0)
                    continue;

                if (grad_list[0] == null)
                    continue;

                int index = i;
                if (kvstore != null)
                {
                    string name = param_names[index];
                    if (grad_list[0].stype == StorageStype.Default && arg_list[0].stype == StorageStype.Default)
                    {
                        kvstore.PushPull(name, grad_list, @out: arg_list, priority: -index);
                    }
                    else
                    {
                        kvstore.Push(name, grad_list, -index);
                        kvstore.Pull(name, arg_list, -index);
                    }
                }

                for(int j = 0; j< arg_list.Length;j++)
                {
                    var w = arg_list[j];
                    var g = grad_list[j];
                    updates[i].Add((index * num_device + j, w, g));
                }

                foreach (var dev_updates in updates.Values)
                {
                    foreach (var item in dev_updates)
                    {
                        var (idx, w, g) = item;
                        updater.Call(idx, w, g);
                    }
                }
            }
        }

        public static void SaveCheckpoint(string prefix, int epoch, Symbol symbol, NDArrayDict arg_params,
            NDArrayDict aux_params, bool remove_amp_cast = true)
        {
            if (symbol != null)
                symbol.Save($"{prefix}-symbol.json", remove_amp_cast);

            NDArrayDict save_dict = new NDArrayDict();
            foreach (var item in arg_params)
            {
                save_dict.Add($"arg:{item.Key}", item.Value);
            }

            foreach (var item in aux_params)
            {
                save_dict.Add($"aux:{item.Key}", item.Value);
            }

            string param_name = $"{prefix}-{epoch.ToString("D4")}.params";
            NDArray.Save(param_name, save_dict);
            Logger.Info($"Saved checkpoint to \"{param_name}\"");
        }

        public static (NDArrayDict, NDArrayDict) LoadParams(string prefix, int epoch)
        {
            var save_dict = NDArray.Load(String.Format("%s-%04d.params", prefix, epoch));
            var arg_params = new NDArrayDict();
            var aux_params = new NDArrayDict();
            if (save_dict != null)
            {
                Logger.Warning($"Params file '{String.Format("%s-%04d.params", prefix, epoch)}' is empty");
                return (arg_params, aux_params);
            }

            string param_name = $"{prefix}-{epoch.ToString("D4")}.params";

            foreach (var item in save_dict)
            {
                if (item.Key.StartsWith("arg:"))
                    arg_params.Add(item.Key.Replace("arg:", ""), item.Value);
                else if (item.Key.StartsWith("aux:"))
                    aux_params.Add(item.Key.Replace("aux:", ""), item.Value);
                else
                    Logger.Warning($"Params file '{param_name}' contains unknown param '{item.Key}'");
            }

            return (arg_params, aux_params);
        }

        public static (Symbol, NDArrayDict, NDArrayDict) LoadCheckpoint(string prefix, int epoch)
        {
            var symbol = Symbol.FromJSON(String.Format("%s-symbol.json", prefix));
            var (arg_params, aux_params) = LoadParams(prefix, epoch);
            return (symbol, arg_params, aux_params);
        }
    }
}