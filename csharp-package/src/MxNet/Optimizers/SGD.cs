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
using MxNet.Numpy;
using System;
using System.Linq;

namespace MxNet.Optimizers
{
    public class SGD : Optimizer
    {
        public readonly bool lazy_update;
        public readonly float momentum;

        public SGD(float learning_rate= 0.01f, float momentum = 0, bool lazy_update = true, bool multi_precision = false, bool use_fused_step = true)
            : base(learning_rate: learning_rate, multi_precision: multi_precision, use_fused_step: use_fused_step)
        {
            this.momentum = momentum;
            this.lazy_update = lazy_update;
            AggregateNum = string.IsNullOrEmpty(Environment.GetEnvironmentVariable("MXNET_OPTIMIZER_AGGREGATION_SIZE"))
                ? 4
                : Convert.ToInt32(Environment.GetEnvironmentVariable("MXNET_OPTIMIZER_AGGREGATION_SIZE"));
        }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            NDArray m = null;
            if (momentum != 0)
            {
                var stype = lazy_update ? weight.stype : StorageStype.Default;
                m = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            }

            return new NDArrayDict {{"mom", m}};
        }

        public override void Update(int[] indices, NDArrayList weights, NDArrayList grads, NDArrayDict[] states)
        {
            _update_impl(indices, weights, grads, states.Select(x=>(ValueTuple.Create<NDArrayDict, ndarray>(x, null))).ToArray());
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            this.UpdateCount(index);
            var lr = this.GetLr(index);
            var wd = this.GetWd(index);
            // preprocess grad
            grad *= this.RescaleGrad;
            if (this.ClipGradient != null)
            {
                grad = nd.Clip(grad, -this.ClipGradient.Value, this.ClipGradient.Value);
            }

            grad += wd * weight;
            // update mom
            if (state["mom"] != null)
            {
                state["mom"] *= this.momentum;
                state["mom"] -= lr * grad;
            }
            else
            {
                state["mom"] = -lr * grad;
            }

            // update weight
            weight += state["mom"];
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            _update_impl(new[] {index}, weight, grad, new (NDArrayDict, ndarray)[] { (state, null) });
        }

        public override void UpdateMultiPrecision(int[] indices, NDArrayList weights, NDArrayList grads, (NDArrayDict, ndarray)[] states)
        {
            var use_multi_precision = MultiPrecision && weights[0].dtype.Name == DType.Float16.Name;
            _update_impl(indices, weights, grads, states, use_multi_precision);
        }

        public override void UpdateMultiPrecision(int index, ndarray weight, ndarray grad, (NDArrayDict, ndarray) state)
        {
            var use_multi_precision = MultiPrecision && weight.dtype.Name == DType.Float16.Name;
            _update_impl(new[] { index }, weight, grad, new (NDArrayDict, ndarray)[] { (state) }, use_multi_precision);
        }

        private void _update_impl(int[] indices, NDArrayList weights, NDArrayList grads, (NDArrayDict, ndarray)[] states,
            bool multi_precision = false)
        {
            var aggregate = true;
            weights.Zip(grads, (weight, grad) =>
            {
                aggregate = aggregate && weight.stype == StorageStype.Default && grad.stype == StorageStype.Default;
                return 0;
            });

            UpdateCount(indices);
            var lrs = GetLrs(indices);
            var wds = GetWds(indices);

            if (aggregate)
            {
                if (!multi_precision)
                {
                    if (momentum > 0)
                        weights = nd.MultiSgdMomUpdate(FlattenList(weights, grads, states), lrs,
                            wds, momentum, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1,
                            weights.Length, weights);
                    else
                        weights = nd.MultiSgdUpdate(FlattenList(weights, grads), lrs,
                            wds, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1, weights.Length, weights);
                }
                else
                {
                    if (momentum > 0)
                        weights = nd.MultiMpSgdMomUpdate(FlattenList(weights, grads, states),
                            lrs, wds, momentum, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1,
                            weights.Length, weights);
                    else
                        weights = nd.MultiMpSgdUpdate(FlattenList(weights, grads, states), lrs,
                            wds, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1, weights.Length, weights);
                }
            }
            else
            {
                for (var i = 0; i < indices.Length; i++)
                {
                    var weight = weights[i];
                    var grad = grads[i];
                    var state = states[i];
                    var lr = lrs[i];
                    var wd = wds[i];

                    if (!multi_precision)
                    {
                        if (state.Item1["mom"] != null)
                            weights[i] = nd.SgdMomUpdate(weight, grad, state.Item1["mom"], lr, momentum, wd, RescaleGrad,
                                ClipGradient.HasValue ? ClipGradient.Value : -1, lazy_update);
                        else
                            weights[i] = nd.SgdMomUpdate(weight, grad, null, lr, momentum, wd, RescaleGrad,
                                ClipGradient.HasValue ? ClipGradient.Value : -1, lazy_update);
                    }
                    else
                    {
                        if (state.Item1["mom"] != null)
                            weights[i] = nd.MpSgdMomUpdate(weight, grad, state.Item1["mom"], state.Item2, lr,
                                momentum, wd, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1,
                                lazy_update);
                        else
                            weights[i] = nd.MpSgdMomUpdate(weight, grad, null, state.Item2, lr, momentum, wd, RescaleGrad,
                                ClipGradient.HasValue ? ClipGradient.Value : -1, lazy_update);
                    }
                }
            }
        }
    }
}