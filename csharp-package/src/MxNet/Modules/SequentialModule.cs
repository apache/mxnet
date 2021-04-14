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
using MxNet.Initializers;
using MxNet.IO;
using MxNet.Metrics;
using MxNet.Optimizers;

namespace MxNet.Modules
{
    public class SequentialModule : BaseModule
    {
        public const string META_TAKE_LABELS = "take_labels";
        public const string META_AUTO_WIRING = "auto_wiring";
        private DataDesc[] _data_shapes;
        private DataDesc[] _label_shapes;
        private string[] _meta_keys;
        private readonly Dictionary<int, (bool?, bool?)> _metas;

        private readonly List<Module> _modules;

        public SequentialModule(Logger logging = null)
        {
            _modules = new List<Module>();
            _metas = new Dictionary<int, (bool?, bool?)>();
            _label_shapes = null;
            _data_shapes = null;
            _meta_keys = new[] {META_TAKE_LABELS, META_AUTO_WIRING};
        }

        public override string[] DataNames
        {
            get
            {
                if (_modules.Count > 0)
                    return _modules[0].DataNames;

                return new string[0];
            }
        }

        public override string[] OutputNames
        {
            get
            {
                if (_modules.Count > 0)
                    return _modules.Last().OutputNames;

                return new string[0];
            }
        }

        public override DataDesc[] DataShapes
        {
            get
            {
                if (Binded)
                    return _modules.First().DataShapes;

                return new DataDesc[0];
            }
        }

        public override DataDesc[] LabelShapes
        {
            get
            {
                if (Binded)
                    return _label_shapes;

                return new DataDesc[0];
            }
        }

        public override string[] LabelNames
        {
            get
            {
                if (_modules.Count > 0)
                    return _modules[0].LabelNames;

                return new string[0];
            }
        }

        public override DataDesc[] OutputShapes
        {
            get
            {
                if (Binded)
                    return _modules.Last().OutputShapes;

                return new DataDesc[0];
            }
        }

        public void Add(Module module, bool? take_labels = null, bool? auto_wiring = null)
        {
            _modules.Add(module);
            _metas.Add(_modules.Count - 1, (take_labels, auto_wiring));
            Binded = false;
            ParamsInitialized = false;
            OptimizerInitialized = false;
        }

        public override void Backward(NDArrayList out_grads = null)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            for (var i = _modules.Count - 1; i >= 0; i--)
            {
                var module = _modules[i];
                module.Backward(out_grads);
                out_grads = module.GetInputGrads()[0];
            }
        }

        public override void Bind(DataDesc[] data_shapes, DataDesc[] label_shapes = null, bool for_training = true,
            bool inputs_need_grad = false, bool force_rebind = false, Module shared_module = null,
            OpGradReq grad_req = OpGradReq.Write)
        {
            if (Binded && !force_rebind)
            {
                Logger.Warning("Already bound, ignoring bind()");
                return;
            }

            if (InputsNeedGrad)
                if (!ForTraining)
                    throw new Exception("Invalid argument! For Training should be true");

            if (shared_module != null)
                throw new Exception("Shared module is not supported");

            if (_modules.Count == 0)
                throw new Exception("Attempting to bind an empty SequentialModule");

            Binded = true;
            _label_shapes = label_shapes;
            var my_data_shapes = data_shapes;

            var anybody_ever_needs_label = false;
            for (var i_layer = 0; i_layer < _modules.Count; i_layer++)
            {
                var module = _modules[i_layer];
                DataDesc[] my_label_shapes = null;
                var (meta_takelabel, meta_autowiring) = _metas[i_layer];
                if (meta_takelabel.HasValue && meta_autowiring.HasValue)
                {
                    my_label_shapes = label_shapes;
                    anybody_ever_needs_label = true;
                }

                var my_inputs_need_grad = inputs_need_grad || ForTraining && i_layer > 0;
                if (!meta_autowiring.HasValue)
                    meta_autowiring = false;

                string[] data_names = null;
                if (meta_autowiring.Value)
                {
                    data_names = module.DataNames;
                    if (data_names.Length != my_data_shapes.Length)
                        throw new Exception("data_names and my_data_shapes are not same length");

                    my_data_shapes = data_names.Zip(my_data_shapes,
                        (new_name, shape) =>
                        {
                            return new DataDesc(new_name, shape.Shape, shape.DataType, shape.Layout);
                        }).ToArray();
                }

                module.Bind(my_data_shapes, my_label_shapes, ForTraining, my_inputs_need_grad, force_rebind, null,
                    grad_req);

                my_data_shapes = module.OutputShapes;
                if (!anybody_ever_needs_label)
                    _label_shapes = null;
            }
        }

        public override void Forward(DataBatch data_batch, bool is_train = true)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            data_batch = data_batch.Shallowcopy();
            for (var i_layer = 0; i_layer < _modules.Count; i_layer++)
            {
                var module = _modules[i_layer];
                module.Forward(data_batch, is_train);
                if (i_layer + 1 == _modules.Count)
                    break;

                data_batch.Data = module.GetOutputs()[0];
                if (data_batch.ProvideData != null)
                {
                    var data_names = module.OutputShapes.Select(x => x.Name).ToArray();
                    if (data_names.Length != data_batch.Data.Length)
                        throw new Exception("data_names and data length not same");

                    data_batch.ProvideData = data_names
                        .Zip(data_batch.Data, (name, x) => { return new DataDesc(name, x.Shape); }).ToArray();
                }
            }
        }

        public override List<NDArrayList> GetInputGrads(bool merge_multi_context = true)
        {
            if (!Binded && !ParamsInitialized && !InputsNeedGrad)
                throw new Exception("Module not binded or param initialized or InputsNeedGrad=false");

            return _modules.First().GetOutputs(merge_multi_context);
        }

        public override List<NDArrayList> GetOutputs(bool merge_multi_context = true)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            return _modules.Last().GetOutputs(merge_multi_context);
        }

        public override (NDArrayDict, NDArrayDict) GetParams()
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            var arg_params = new NDArrayDict();
            var aux_params = new NDArrayDict();

            foreach (var module in _modules)
            {
                var (arg, aux) = module.GetParams();
                arg_params.Add(arg);
                aux_params.Add(aux);
            }

            return (arg_params, aux_params);
        }

        public override void InitOptimizer(string kvstore = "local", Optimizer optimizer = null,
            Dictionary<string, object> optimizer_params = null, bool force_init = false)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            if (OptimizerInitialized && !force_init)
            {
                Logger.Warning("optimizer already initialized, ignoring.");
                return;
            }

            foreach (var module in _modules) module.InitOptimizer(kvstore, optimizer, optimizer_params, force_init);

            OptimizerInitialized = true;
        }

        public override void InitParams(Initializer initializer = null, NDArrayDict arg_params = null,
            NDArrayDict aux_params = null, bool allow_missing = false, bool force_init = false,
            bool allow_extra = false)
        {
            if (ParamsInitialized && !force_init)
                return;

            if (!Binded)
                throw new Exception("call bind before initializing the parameters");

            foreach (var module in _modules)
                module.InitParams(initializer, arg_params, aux_params, allow_missing, force_init, allow_extra);

            void _check_name(Dictionary<string, int> known_names, string[] new_names, Module[] modules, int i)
            {
                foreach (var name in new_names)
                {
                    if (known_names.ContainsKey(name))
                        throw new Exception("Duplicate parameter names: " +
                                            $"name '{name}' in layer {i} ({modules[i].GetType().Name}) is already " +
                                            $"used in layer {known_names[name]} ({modules[known_names[name]]}");

                    known_names[name] = i;
                }
            }

            var arg_names = new Dictionary<string, int>();
            var aux_names = new Dictionary<string, int>();
            for (var i_layer = 0; i_layer < _modules.Count; i_layer++)
            {
                var (arg_p, aux_p) = _modules[i_layer + i_layer].GetParams();
                _check_name(arg_names, arg_p.Keys.ToArray(), _modules.ToArray(), i_layer);
                _check_name(aux_names, aux_p.Keys.ToArray(), _modules.ToArray(), i_layer);
            }

            ParamsInitialized = true;
        }

        public override void InstallMonitor(Monitor mon)
        {
            if (!Binded)
                throw new Exception("Module not binded");

            foreach (var module in _modules) module.InstallMonitor(mon);
        }

        public override void Update()
        {
            if (!Binded && !ParamsInitialized && !OptimizerInitialized)
                throw new Exception("Module not binded or param initialized oo optimizer initialized");

            foreach (var module in _modules) module.Update();
        }

        public override void UpdateMetric(EvalMetric eval_metric, NDArrayList labels, bool pre_sliced = false)
        {
            if (!Binded && !ParamsInitialized)
                throw new Exception("Module not binded and param initialized");

            var i_layer = 0;
            foreach (var module in _modules)
            {
                var (meta_takelabel, meta_autowiring) = _metas[i_layer];
                if (meta_takelabel.HasValue && meta_autowiring.HasValue)
                    module.UpdateMetric(eval_metric, labels, pre_sliced);

                i_layer++;
            }
        }
    }
}