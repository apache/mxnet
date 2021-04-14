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
using MxNet.Initializers;
using MxNet.IO;
using MxNet.Metrics;
using MxNet.Optimizers;
using System.Linq;

namespace MxNet.Modules
{
    public class BucketingModule : BaseModule
    {
        private readonly Func<int, (Symbol, string[], string[])> _sym_Gen;
        private readonly int _default_bucket_key;
        private Dictionary<string, object> _compression_params;
        private string[] _fixed_param_names;
        private string[] _state_names;
        private Context[] _context;
        private int[] _work_load_list;
        private Dictionary<string, Context>[] _group2ctxs;
        public Dictionary<int, Module> _buckets;
        public Module _curr_module;
        private int? _curr_bucket_key;
        private bool _params_dirty;
        private Monitor _monitor;
        private OpGradReq _grad_req;

        public BucketingModule(Func<int, (Symbol, string[], string[])> sym_gen, int default_bucket_key,
            Logger logging = null,
            Context[] context = null, int[] work_load_list = null, string[] fixed_param_names = null,
            string[] state_names = null,
            Dictionary<string, Context>[] group2ctxs = null, Dictionary<string, object> compression_params = null)
        {
            _sym_Gen = sym_gen;
            _default_bucket_key = default_bucket_key;
            var (symbol, data_names, label_names) = CallSymGen(default_bucket_key);
            data_names = data_names ?? new string[0];
            label_names = label_names ?? new string[0];
            state_names = state_names ?? new string[0];
            fixed_param_names = fixed_param_names ?? new string[0];

            CheckInputNames(symbol, data_names, "data", true);
            CheckInputNames(symbol, label_names, "label", false);
            CheckInputNames(symbol, state_names, "label", true);
            CheckInputNames(symbol, fixed_param_names, "fixed_param", true);

            _compression_params = compression_params;
            _fixed_param_names = fixed_param_names;
            _state_names = state_names;
            _context = context;
            _work_load_list = work_load_list;
            _group2ctxs = group2ctxs;

            _buckets = new Dictionary<int, Module>();
            _curr_module = null;
            _curr_bucket_key = null;
            _params_dirty = false;
            _monitor = null;
            _grad_req = OpGradReq.Null;
        }

        public override string[] DataNames
        {
            get
            {
                if (Binded)
                    return _curr_module.DataNames;

                var (_, data_names, _) = _sym_Gen(_default_bucket_key);
                return data_names;
            }
        }

        public override string[] OutputNames
        {
            get
            {
                if (Binded)
                    return _curr_module.DataNames;

                var (symbol, _, _) = _sym_Gen(_default_bucket_key);
                return symbol.ListOutputs().ToArray();
            }
        }

        public override string[] LabelNames
        {
            get
            {
                if (Binded)
                    return _curr_module.LabelNames;

                var (_, _, labal_names) = _sym_Gen(_default_bucket_key);
                return labal_names;
            }
        }

        public override DataDesc[] DataShapes
        {
            get
            {
                if (Binded)
                    return _curr_module.DataShapes;

                throw new Exception("Module is not bound");
            }
        }

        public override DataDesc[] LabelShapes
        {
            get
            {
                if (Binded)
                    return _curr_module.LabelShapes;

                throw new Exception("Module is not bound");
            }
        }

        public override DataDesc[] OutputShapes
        {
            get
            {
                if (Binded)
                    return _curr_module.OutputShapes;

                throw new Exception("Module is not bound");
            }
        }

        public override Symbol Symbol
        {
            get
            {
                if (Binded)
                    return _curr_module.Symbol;

                throw new Exception("Module is not bound");
            }
        }

        private void ResetBind()
        {
            Binded = false;
            _buckets = new Dictionary<int, Module>();
            _curr_module = null;
            _curr_bucket_key = null;
        }

        private (Symbol, string[], string[]) CallSymGen(int bucketKey)
        {
            return _sym_Gen(bucketKey);
        }


        public override void Backward(NDArrayList out_grads = null)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");
            _curr_module.Backward(out_grads);
        }

        public override void Bind(DataDesc[] data_shapes, DataDesc[] label_shapes = null, bool for_training = true,
            bool inputs_need_grad = false, bool force_rebind = false, Module shared_module = null,
            OpGradReq grad_req = OpGradReq.Write)
        {
            NDArrayDict arg_params = null;
            NDArrayDict aux_params = null;

            if (ParamsInitialized)
                (arg_params, aux_params) = GetParams();

            if (force_rebind)
                ResetBind();

            if (Binded)
            {
                Logger.Warning("Already bound, ignoring Bind()");
                return;
            }

            if (shared_module != null)
                throw new NotSupportedException("shared_module for BucketingModule is not supported");

            ForTraining = for_training;
            InputsNeedGrad = inputs_need_grad;
            Binded = true;
            _grad_req = grad_req;

            var (symbol, data_names, label_names) = _sym_Gen(_default_bucket_key);
            var module = new Module(symbol, data_names, label_names, context: _context, work_load_list: _work_load_list,
                                fixed_param_names: _fixed_param_names, state_names: _state_names, group2ctxs: _group2ctxs,
                                compression_params: _compression_params);
            module.Bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                        force_rebind: false, shared_module: null, grad_req: _grad_req);
            _curr_module = module;
            _curr_bucket_key = _default_bucket_key;
            _buckets[_default_bucket_key] = module;
            if (ParamsInitialized)
                SetParams(arg_params, aux_params);
        }

        public override void Forward(DataBatch data_batch, bool is_train = true)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            SwitchBucket(data_batch.BucketKey.Value, data_batch.ProvideData, data_batch.ProvideLabel);
        }

        public override List<NDArrayList> GetInputGrads(bool merge_multi_context = true)
        {
            if (!Binded && !ParamsInitialized && !InputsNeedGrad)
                throw new Exception("Module is not bound or initialized or InputsNeedGrad=false");

            return _curr_module.GetInputGrads(merge_multi_context);
        }

        public override List<NDArrayList> GetOutputs(bool merge_multi_context = true)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            return _curr_module.GetOutputs(merge_multi_context);
        }

        public override (NDArrayDict, NDArrayDict) GetParams()
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");
            _curr_module._params_dirty = _params_dirty;
            _params_dirty = false;
            return _curr_module.GetParams();
        }

        public override void InitOptimizer(string kvstore = "local", Optimizer optimizer = null,
            Dictionary<string, object> optimizer_params = null, bool force_init = false)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            if(OptimizerInitialized && !force_init)
            {
                Logger.Warning("Optimizer already initialized, ignoring.");
                return;
            }

            _curr_module.InitOptimizer(kvstore, optimizer, optimizer_params, force_init);
            OptimizerInitialized = true;
        }

        public override void InitParams(Initializer initializer = null, NDArrayDict arg_params = null,
            NDArrayDict aux_params = null, bool allow_missing = false, bool force_init = false,
            bool allow_extra = false)
        {
            if (ParamsInitialized && !force_init)
                return;

            if (!Binded)
                throw new Exception("Call Bind before initializing the parameters");

            _curr_module.InitParams(initializer, arg_params, aux_params, allow_missing, force_init, allow_extra);
            _params_dirty = false;
            ParamsInitialized = true;
        }

        public override void InstallMonitor(Monitor mon)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            foreach (var mod in _buckets.Values)
            {
                mod.InstallMonitor(mon);
            }
        }

        public override void Update()
        {
            if (!Binded && !ParamsInitialized && !OptimizerInitialized)
                throw new Exception("Module is not bound or initialized");

            _params_dirty = true;
            _curr_module.Update();
        }

        public override void UpdateMetric(EvalMetric eval_metric, NDArrayList labels, bool pre_sliced = false)
        {
            if (!Binded && !ParamsInitialized && !OptimizerInitialized)
                throw new Exception("Module is not bound or initialized");

            _curr_module.UpdateMetric(eval_metric, labels, pre_sliced);
        }

        public override void SetParams(NDArrayDict arg_params = null, NDArrayDict aux_params = null,
            bool allow_missing = false, bool force_init = false, bool allow_extra = false)
        {
            if(!allow_missing)
            {
                InitParams(null, arg_params, aux_params, allow_missing, force_init);
                return;
            }

            if(ParamsInitialized && !force_init)
            {
                Logger.Warning("Parameters already initialized and force_init=False. " +
                                    "set_params call ignored.");
                return;
            }

            _curr_module.SetParams(arg_params, aux_params, allow_missing, force_init, allow_extra);
        }

        public override List<NDArrayList> GetStates(bool merge_multi_context = false)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            return _curr_module.GetStates(merge_multi_context);
        }

        public override void SetStates(List<NDArrayList> states, int value)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            _curr_module.SetStates(states, value);
        }

        public void SwitchBucket(int bucket_key, DataDesc[] data_shapes, DataDesc[] label_shapes = null)
        {
            if (!Binded)
                throw new Exception("Call Bind before switching bucket");

            if(!_buckets.ContainsKey(bucket_key))
            {
                var (symbol, data_names, label_names) = _sym_Gen(bucket_key);
                var module = new Module(symbol, data_names, label_names,
                                        context: _context,
                                        work_load_list: _work_load_list,
                                        fixed_param_names: _fixed_param_names,
                                        state_names: _state_names,
                                        group2ctxs: _group2ctxs,
                                        compression_params: _compression_params);
                module.Bind(data_shapes, label_shapes, _curr_module.ForTraining,
                        _curr_module.InputsNeedGrad,
                        force_rebind: false, shared_module: _buckets[_default_bucket_key],
                        grad_req: _grad_req);

                if (_monitor != null)
                    module.InstallMonitor(_monitor);

                _buckets[bucket_key] = module;
            }

            _curr_module = _buckets[bucket_key];
            _curr_bucket_key = bucket_key;
        }

        public override void Prepare(DataBatch data_batch, Func<DataBatch, NDArrayDict> sparse_row_id_fn = null)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module is not bound or initialized");

            var bucket_key = data_batch.BucketKey;
            var original_bucket_key = _curr_bucket_key.Value;
            var data_shapes = data_batch.ProvideData;
            var label_shapes = data_batch.ProvideLabel;
            SwitchBucket(bucket_key.Value, data_shapes, label_shapes);
            _curr_module.Prepare(data_batch, sparse_row_id_fn: sparse_row_id_fn);
            SwitchBucket(original_bucket_key, null, null);
        }
    }
}