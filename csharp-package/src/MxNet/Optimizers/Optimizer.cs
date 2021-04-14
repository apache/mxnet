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
using MxNet.Gluon;
using MxNet.Numpy;

namespace MxNet.Optimizers
{
    public class OptimState : NDArrayDict
    {
    }

    public abstract class Optimizer
    {
        internal Dictionary<int, Dictionary<int, int>> all_index_update_counts =
            new Dictionary<int, Dictionary<int, int>>();

        internal Dictionary<int, int> index_update_count = new Dictionary<int, int>();

        private float lr;
        private Dictionary<string, float> lr_mult = new Dictionary<string, float>();

        private Dictionary<string, Optimizer> opt_registry = new Dictionary<string, Optimizer>();
        private readonly (Dictionary<string, Dictionary<string, string>>, List<string>) sym_info;
        private Dictionary<string, float> wd_mult = new Dictionary<string, float>();
        private bool use_fused_step;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Optimizer" /> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="name">The name.</param>
        public Optimizer(float rescale_grad = 1, Dictionary<int, string> param_idx2name = null, float wd = 0,
            float? clip_gradient = null, float learning_rate = 0.01f, LRScheduler lr_scheduler = null,
            Symbol sym = null, uint begin_num_update = 0, bool multi_precision = false, bool use_fused_step = true,
            Dictionary<int, Parameter> param_dict = null)
        {
            lr = learning_rate;
            RescaleGrad = rescale_grad;
            Scheduler = lr_scheduler;
            this.use_fused_step = use_fused_step;
            if (Scheduler != null)
                Scheduler.BaseLearningRate = learning_rate;

            WD = wd;
            BeginNumUpdate = begin_num_update;
            NumUpdate = begin_num_update;
            all_index_update_counts.Add(0, new Dictionary<int, int>());
            index_update_count = all_index_update_counts[0];
            ClipGradient = clip_gradient;
            MultiPrecision = multi_precision;
            AggregateNum = 0;
            if (param_idx2name == null)
                param_idx2name = new Dictionary<int, string>();

            Idx2Name = param_idx2name;
            if (sym != null)
                sym_info = (sym.ListAttributeDict(), sym.ListArguments().ToList());
            else
                sym_info = new ValueTuple<Dictionary<string, Dictionary<string, string>>, List<string>>(
                    new Dictionary<string, Dictionary<string, string>>(), new List<string>());

            if (param_dict != null)
                ParamDict = param_dict;
            else
                ParamDict = new Dictionary<int, Parameter>();

            SetLrMult(new Dictionary<string, float>());
            SetWdMult(new Dictionary<string, float>());
        }

        public float LearningRate
        {
            get
            {
                if (Scheduler != null)
                    return Scheduler.Call(NumUpdate);
                return lr;
            }
            set
            {
                lr = value;
            }
        }

        public float WD { get; set; }
        public float? ClipGradient { get; set; }
        public float RescaleGrad { get; set; }
        public LRScheduler Scheduler { get; set; }
        public bool MultiPrecision { get; set; }
        public uint BeginNumUpdate { get; set; }
        public uint NumUpdate { get; set; }
        public int AggregateNum { get; set; }
        public Dictionary<int, string> Idx2Name { get; set; }
        public Dictionary<int, Parameter> ParamDict { get; set; }

        public abstract NDArrayDict CreateState(int index, ndarray weight);

        public virtual (NDArrayDict, ndarray) CreateStateMultiPrecision(int index, ndarray weight)
        {
            ndarray weight_master_copy = null;
            if (MultiPrecision && weight.dtype.Name == DType.Float16.Name)
            {
                weight_master_copy = weight.AsType(DType.Float32);
                return (CreateState(index, weight_master_copy), weight_master_copy);
            }

            if (!MultiPrecision && weight.dtype.Name == DType.Float16.Name)
                Logger.Warning("Accumulating with float16 in optimizer can lead to " +
                               "poor accuracy or slow convergence. " +
                               "Consider using multi_precision=True option of the " +
                               "optimizer");

            return (CreateState(index, weight), weight);
        }

        public virtual void Update(int[] indices, NDArrayList weights, NDArrayList grads, NDArrayDict[] states)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                if(use_fused_step)
                    FusedStep(indices[i], weights[i], grads[i], states[i]);
                else
                    Step(indices[i], weights[i], grads[i], states[i]);
            }
        }

        public abstract void Step(int index, ndarray weight, ndarray grad, NDArrayDict state);

        public abstract void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state);

        public virtual void UpdateMultiPrecision(int[] indices, NDArrayList weights, NDArrayList grads, (NDArrayDict, ndarray)[] states)
        {
            for(int i = 0;i<indices.Length;i++)
            {
                UpdateMultiPrecision(indices[i], weights[i], grads[i], states[i]);
            }
        }

        public virtual void UpdateMultiPrecision(int index, ndarray weight, ndarray grad, (NDArrayDict, ndarray) state)
        {
            if (MultiPrecision && weight.dtype.Name == DType.Float16.Name)
            {
                var weight_master_copy = state.Item2;
                var grad32 = grad.AsType(DType.Float32);
                if (use_fused_step)
                    FusedStep(index, weight_master_copy, grad32, state.Item1);
                else
                    Step(index, weight_master_copy, grad32, state.Item1);

                weight_master_copy.Cast(weight.dtype).CopyTo(weight);
            }
            else
            {
                if(use_fused_step)
                    FusedStep(index, weight, grad, state.Item1);
                else
                    Step(index, weight, grad, state.Item1);
            }
        }

        public void SetLearningRate(float lr)
        {
            Logger.Warning("[DEPRECATED] Sets lr scale. Use SetLrMult instead");
        }

        public static Updater GetUpdater(Optimizer optimizer)
        {
            return optimizer.GetUpdater();
        }

        internal void SetLrMult(Dictionary<string, float> args_lr_mult)
        {
            lr_mult = new Dictionary<string, float>();
            if (sym_info.Item1.Count > 0)
            {
                var (attr, arg_names) = sym_info;
                foreach (var name in arg_names)
                    if (attr.ContainsKey(name) && attr[name].ContainsKey("__lr_mult__"))
                        if (float.TryParse(attr[name]["__lr_mult__"], out var attrValue))
                            lr_mult[name] = attrValue;
            }

            foreach (var item in args_lr_mult) lr_mult[item.Key] = item.Value;
        }

        internal void SetWdMult(Dictionary<string, float> args_wd_mult)
        {
            wd_mult = new Dictionary<string, float>();
            foreach (var n in Idx2Name.Values)
                if (!n.EndsWith("_weight") || n.EndsWith("_gamma"))
                    wd_mult[n] = 0;

            if (sym_info.Item1.Count > 0)
            {
                var (attr, arg_names) = sym_info;
                foreach (var name in arg_names)
                    if (attr.ContainsKey(name) && attr[name].ContainsKey("__wd_mult__"))
                        if (float.TryParse(attr[name]["__wd_mult__"], out var attrValue))
                            wd_mult[name] = attrValue;
            }

            foreach (var item in args_wd_mult) wd_mult[item.Key] = item.Value;
        }

        internal void SetCurrentContext(int device_id)
        {
            if (all_index_update_counts.ContainsKey(device_id))
                all_index_update_counts[device_id] = new Dictionary<int, int>();

            index_update_count = all_index_update_counts[device_id];
        }

        internal void UpdateCount(params int[] index)
        {
            foreach (var idx in index)
            {
                if (!index_update_count.ContainsKey(idx))
                    index_update_count[idx] = (int) BeginNumUpdate;

                index_update_count[idx] += 1;
                NumUpdate = (uint) Math.Max(index_update_count[idx], NumUpdate);
            }
        }

        public virtual float[] GetLrs(int[] indices)
        {
            float lr = 0;
            if (Scheduler != null)
                lr = Scheduler.Call(NumUpdate);
            else
                lr = LearningRate;

            var lrs = new float[indices.Length];
            for (var i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                lrs[i] = lr;
                if (ParamDict.ContainsKey(index))
                {
                    lrs[i] *= ParamDict[index].Lr_Mult;
                }
                else if (lr_mult.ContainsKey(index.ToString()))
                {
                    lrs[i] *= lr_mult[index.ToString()];
                }
                else if (Idx2Name.ContainsKey(index))
                {
                    float Idx2Name_lrvalue = 1;
                    if (float.TryParse(Idx2Name[index], out Idx2Name_lrvalue)) lrs[i] *= Idx2Name_lrvalue;
                }
            }

            return lrs;
        }

        public virtual float GetLr(int index)
        {
            return GetLrs(new[] {index})[0];
        }

        public virtual float[] GetWds(int[] indices)
        {
            var wds = new float[indices.Length];
            for (var i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                if (ParamDict.ContainsKey(index))
                {
                    wds[i] *= ParamDict[index].Wd_Mult;
                }
                else if (wd_mult.ContainsKey(index.ToString()))
                {
                    wds[i] *= wd_mult[index.ToString()];
                }
                else if (Idx2Name.ContainsKey(index))
                {
                    float Idx2Name_lrvalue = 1;
                    if (float.TryParse(Idx2Name[index], out Idx2Name_lrvalue)) wds[i] *= Idx2Name_lrvalue;
                }
            }

            return wds;
        }

        public virtual float GetWd(int index)
        {
            return GetWds(new[] {index})[0];
        }

        internal static NDArrayList FlattenList(NDArrayList weights, NDArrayList grads, (NDArrayDict, ndarray)[] states = null)
        {
            var result = new NDArrayList();
            for (int i = 0; i < weights.Length; i++)
            {
                result.Add(weights[i]);
                result.Add(grads[i]);
                if(states != null)
                {
                    var state = states[i];
                    if (state.Item1 != null)
                        result.Add(state.Item1.Values.ToArray());

                    if (state.Item2 != null)
                        result.Add(state.Item2);
                }
            }

            return result.ToArray();
        }

        public Updater GetUpdater()
        {
            return new Updater(this);
        }
    }
}
