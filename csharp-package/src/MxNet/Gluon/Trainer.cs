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
using System.IO;
using System.Linq;
using MxNet.Initializers;
using MxNet.KVstore;
using MxNet.Numpy;
using MxNet.Optimizers;

namespace MxNet.Gluon
{
    public class Trainer
    {
        private readonly Dictionary<string, object> _compression_params;
        private readonly bool _contains_sparse_grad;
        private readonly bool _contains_sparse_weight;
        private readonly Context[] _contexts;
        private bool? _distributed;
        private Initializer _init_optimizer;
        internal bool _kv_initialized;
        internal KVStore _kvstore;
        private readonly Dictionary<string, object> _kvstore_params = new Dictionary<string, object>();
        private readonly Dictionary<string, int> _param2idx = new Dictionary<string, int>();
        private readonly List<Parameter> _params;
        internal List<Parameter> _params_to_init;
        private readonly float _scale;
        internal bool? _update_on_kvstore;
        private List<Updater> _updaters;

        public Trainer(ParameterDict @params, Optimizer optimizer, string kvstore = "device",
            Dictionary<string, object> compression_params = null, bool? update_on_kvstore = null)
        {
            var paramValues = @params.Values();
            _params = new List<Parameter>();
            var keys = @params.Keys().ToList();
            keys.Sort();
            for (var i = 0; i < keys.Count; i++)
            {
                var param = @params[keys[i]];
                _param2idx[keys[i]] = i;
                _params.Add(param);
                param.SetTrainer(this);
                if (param.Stype != StorageStype.Default)
                    _contains_sparse_weight = true;

                if (param.Grad_Stype != StorageStype.Default)
                    _contains_sparse_grad = true;
            }

            _compression_params = compression_params;
            _contexts = CheckContexts();
            InitOptimizer(optimizer);
            _scale = optimizer.RescaleGrad;

            if (this.Optimizer.AggregateNum > 1 && update_on_kvstore != null)
            {
                if (update_on_kvstore.HasValue)
                {
                    throw new Exception("Cannot set update_on_kvstore=True when optimizer.aggregate_num > 1.");
                }
            }

            if (update_on_kvstore == null && this.Optimizer.AggregateNum > 1)
            {
                update_on_kvstore = false;
            }

            _kvstore_params = new Dictionary<string, object>();
            _kvstore_params.Add("kvstore", kvstore);
            _kvstore_params.Add("update_on_kvstore", update_on_kvstore);
            _kvstore = null;
            _update_on_kvstore = null;
            _params_to_init = new List<Parameter>();
            ResetKVstore();
        }

        public float LearningRate
        {
            get => Optimizer.LearningRate;
            set => Optimizer.SetLearningRate(value);
        }

        public Optimizer Optimizer { get; private set; }

        internal Context[] CheckContexts()
        {
            var contexts = new List<Context>();
            foreach (var param in _params)
            {
                var ctx = param.ListCtx();
                if (contexts.Count == 0) contexts = ctx.ToList();

                if (contexts[0].ToString() != ctx[0].ToString() || contexts.Count != ctx.Count())
                    throw new Exception("All Parameters must be initialized on the same set of contexts, " +
                                        $"but Parameter {param.Name} is initialized on {ctx[0]} while previous Parameters " +
                                        $"are initialized on {contexts[0]}.");

                contexts = ctx.ToList();
            }

            return contexts.ToArray();
        }

        internal void InitOptimizer(Optimizer optimizer)
        {
            Optimizer = optimizer;
            _updaters = new List<Updater>();
            foreach (var item in _contexts) _updaters.Add(new Updater(optimizer));
        }

        internal void InitParams()
        {
            if (!_kv_initialized)
                throw new Exception("Cannot initialize parameters in KVStore " +
                                    "when KVStore is not initialized.");

            var params_to_init = new List<Parameter>();
            if (_kvstore != null)
                foreach (var param in _params_to_init)
                    if (param.deferred_init != null)
                    {
                        params_to_init.Add(param);
                    }
                    else
                    {
                        var param_arrays = param.CheckAndGet(param._data, null);
                        var idx = _param2idx[param._uuid];
                        _kvstore.Init(idx.ToString(), param_arrays[0]);
                        if (param.Stype == StorageStype.Default)
                            _kvstore.Init(idx, param_arrays[0]);
                        else
                            _kvstore.Broadcast(idx, param_arrays[0], param_arrays);
                    }

            _params_to_init = params_to_init;
        }

        internal void ResetKVstore()
        {
            if (_kvstore != null && _kvstore.Type.Contains("dist"))
                throw new Exception("Cannot reset distributed KVStore.");

            _kv_initialized = false;
            _kvstore = null;
            _distributed = null;
            _update_on_kvstore = null;
            _params_to_init = _params.ToList();
        }

        internal void InitKVstore()
        {
            var config = _kvstore_params;
            var update_on_kvstore = false;
            KVStore kvstore = null;
            if (_contains_sparse_weight)
            {
                (kvstore, update_on_kvstore) = MxModel.CreateSparseKVStore(config["kvstore"].ToString());
                _distributed = kvstore.Type.Contains("dist");
                if (!(bool) config["update_on_kvstore"])
                    throw new Exception("Cannot set update_on_kvstore=False when sparse weights " +
                                        "are present.");
            }
            else if (_contains_sparse_grad)
            {
                var arg_arrays = new NDArrayDict();
                foreach (var param in _params) arg_arrays[param._uuid] = param.Data(_contexts[0]);

                (kvstore, _) = MxModel.CreateKVStore(config["kvstore"].ToString(), _contexts.Length, arg_arrays);
                if (kvstore != null)
                    _distributed = kvstore.Type.Contains("dist");
                else
                    _distributed = false;

                update_on_kvstore = _distributed.Value;
                if (config.ContainsKey("update_on_kvstore"))
                {
                    if ((bool) config["update_on_kvstore"] == false && _distributed.Value)
                        throw new Exception("Cannot set update_on_kvstore=False on dist kvstore " +
                                            "when sparse gradients are present.");

                    update_on_kvstore = (bool) config["update_on_kvstore"];
                }
            }
            else
            {
                var arg_arrays = new NDArrayDict();
                foreach (var param in _params) arg_arrays[param._uuid] = param.Data(_contexts[0]);

                (kvstore, update_on_kvstore) =
                    MxModel.CreateKVStore(config["kvstore"].ToString(), _contexts.Length, arg_arrays);
                if (kvstore != null)
                    _distributed = kvstore.Type.Contains("dist");
                else
                    _distributed = false;

                if (_distributed.Value && kvstore.Type.Contains("async"))
                {
                    update_on_kvstore = true;
                    if (!(bool) config["update_on_kvstore"])
                        throw new Exception("Please set update_on_kvstore=True " +
                                            "when training in async mode.");
                }

                if (config.ContainsKey("update_on_kvstore") && config["update_on_kvstore"] != null)
                    update_on_kvstore = (bool) config["update_on_kvstore"];

                if (update_on_kvstore && !kvstore.IsCapable("optimizer"))
                {
                    if (update_on_kvstore)
                    {
                        throw new Exception($"Please set update_on_kvstore=False when training with {kvstore.GetType().Name}");
                    }
                    update_on_kvstore = false;
                }
            }

            if (kvstore != null)
            {
                if (_compression_params != null)
                    kvstore.SetGradientCompression(_compression_params);

                if (update_on_kvstore)
                    kvstore.SetOptimizer(Optimizer);

                _kvstore = kvstore;
                _update_on_kvstore = update_on_kvstore;
            }
            else
            {
                _kvstore = null;
                _update_on_kvstore = false;
            }

            _kv_initialized = true;
        }

        internal void RowSparsePull(Parameter parameter, NDArrayList @out, ndarray row_id, bool full_idx = false)
        {
            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            var idx = _param2idx[parameter._uuid];
            if (full_idx && _kvstore.Type.Contains("dist"))
            {
                if (row_id.size != @out[0].shape[0])
                    throw new Exception("row_id size not equal to @out row size");

                _kvstore.Pull(idx.ToString(), @out.ToArray(), -idx, false);
            }
            else
            {
                _kvstore.RowSparsePull(idx.ToString(), @out.ToArray(), -idx, row_id);
            }
        }

        internal void CheckAndRescaleGrad(float scale)
        {
            if (_update_on_kvstore.HasValue && _update_on_kvstore.Value && _distributed.HasValue &&
                _distributed.Value && _kv_initialized)
                if (Optimizer.RescaleGrad != scale)
                    Logger.Warning("Possible change in the `batch_size` from previous " +
                                   "`step` detected. Optimizer gradient normalizing " +
                                   "factor will not change w.r.t new batch_size when " +
                                   "update_on_kvstore=True and when distributed kvstore " +
                                   "is used.");
        }

        public void Step(int batch_size, bool ignore_stale_grad = false)
        {
            var rescale_grad = _scale / batch_size;
            CheckAndRescaleGrad(rescale_grad);

            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            AllReduceGrads();
            _update(ignore_stale_grad);
        }

        public void AllReduceGrads()
        {
            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            if (_kvstore == null && _update_on_kvstore.Value)
                throw new Exception("allreduce_grads() when parameters are updated on kvstore " +
                                    "is not supported. Try setting `update_on_kvstore` " +
                                    "to False when creating trainer.");

            _all_reduce_grads();
        }

        public void _all_reduce_grads()
        {
            if (_kvstore == null)
                return;

            var i = 0;
            foreach (var param in _params)
            {
                if (param.GradReg != OpGradReq.Null)
                {
                    var idx = this._param2idx[param._uuid];
                    var grad_list = param.ListGrad();
                    NDArrayList pull_list = null;
                    // sparse gradients, call push and pull separately
                    if (grad_list[0].stype !=  StorageStype.Default)
                    {
                        this._kvstore.Push(idx, grad_list, priority: -i);
                        if (param.Stype == StorageStype.Default)
                        {
                            if (this._update_on_kvstore.Value)
                            {
                                pull_list = param.ListData();
                            }
                            else
                            {
                                pull_list = param.ListGrad();
                            }

                            this._kvstore.Pull(idx, pull_list, priority: -i, ignore_sparse: this._distributed.Value);
                        }
                    }
                    else
                    {
                        // allreduce dense gradients if not update_on_kvstore,
                        // otherwise push dense gradients, pull dense weights
                        if (this._update_on_kvstore.Value)
                        {
                            this._kvstore.PushPull(idx, grad_list, @out: param.ListData(), priority: -i);
                        }
                        else
                        {
                            this._kvstore.PushPull(idx, grad_list, @out: param.ListGrad(), priority: -i);
                        }
                    }
                }

                i++;
            }
        }

        public void Update(int batch_size, bool ignore_stale_grad = false)
        {
            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            if (_kvstore == null && _update_on_kvstore.Value)
                throw new Exception("update() when parameters are updated on kvstore " +
                                    "is not supported. Try setting `update_on_kvstore` " +
                                    "to False when creating trainer.");

            CheckAndRescaleGrad(_scale / batch_size);
            _update(ignore_stale_grad);
        }

        private void _update(bool ignore_stale_grad = false)
        {
            var updates = new List<(int[], NDArrayList, NDArrayList)>();
            var indices = new List<int>();
            var grads = new NDArrayList();
            var arrays = new NDArrayList();

            for (var i = 0; i < _params.Count; i++)
            {
                var param = _params[i];
                if (param.GradReg == OpGradReq.Null)
                    continue;

                if (!ignore_stale_grad)
                {
                    var datalist = param.CheckAndGet(param._data, null);
                    foreach (var data in datalist)
                        if (!data.FreshGrad)
                            Logger.Warning(
                                $"Gradient of Parameter `{param.Name}` on context {data.ctx} has not been updated " +
                                "by backward since last `step`. This could mean a bug in your " +
                                "model that made it only use a subset of the Parameters (Blocks) " +
                                "for this iteration. If you are intentionally only using a subset, " +
                                "call step with ignore_stale_grad=True to suppress this " +
                                "warning and skip updating of Parameters with stale gradient");
                }

                if (_kvstore == null && _update_on_kvstore.Value)
                {
                    continue;
                }


                var data_list = param.ListData();
                var grad_list = param.ListGrad();
                for (int idx = 0; idx < data_list.Length; idx++)
                {
                    indices.Add(i);
                    grads.Add(grad_list[idx]);
                    arrays.Add(data_list[idx]);
                }

                //var ulist = param.ListData().Zip(param.ListGrad(), (arr, grad) =>
                //{
                //    if (!ignore_stale_grad || arr.FreshGrad)
                //    {
                //        indices.Add(i);
                //        grads.Add(grad);
                //        arrays.Add(arr);
                //        arr.FreshGrad = false;
                //    }

                //    return (i, grad, arr);
                //}).ToList();

            }

            foreach (var u in _updaters)
            {
                updates.Add((indices.ToArray(), arrays, grads));
            }

            if (!(_kvstore == null && _update_on_kvstore.Value))
            {
                int ui = 0;
                foreach (var updater in _updaters)
                {
                    var (i, w, g) = updates[ui];
                    updater.Call(i, g, w);
                }
            }
            //_updaters.Zip(updates, (updater, upd) =>
            //{
            //    var (i, w, g) = upd;
            //    updater.Call(i, g, w);
            //    return true;
            //});
        }

        public void SaveStates(string fname)
        {
            if (Optimizer == null)
                return;

            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            if (_update_on_kvstore.Value)
            {
                if (_params_to_init == null)
                    throw new Exception("Cannot save trainer states when some " +
                                        "parameters are not yet initialized in kvstore.");

                _kvstore.SaveOptimizerStates(fname, true);
            }
            else
            {
                var state_str = _updaters[0].GetStates(true);
                File.WriteAllText(fname, state_str);
            }
        }

        public void LoadStates(string fname)
        {
            if (!_kv_initialized)
                InitKVstore();

            if (_params_to_init != null)
                InitParams();

            if (_update_on_kvstore.Value)
            {
                _kvstore.LoadOptimizerStates(fname);
                Optimizer = _kvstore._updater.optimizer;
            }
            else
            {
                var state_str = File.ReadAllText(fname);
                foreach (var updater in _updaters)
                {
                    updater.SetStates(state_str);
                    updater.optimizer = _updaters[0].optimizer;
                }

                Optimizer = _updaters[0].optimizer;
            }

            Optimizer.ParamDict = new Dictionary<int, Parameter>();
            for (var i = 0; i < _params.Count; i++) Optimizer.ParamDict.Add(i, _params[i]);
        }
    }
}