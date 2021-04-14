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

namespace MxNet.Optimizers
{
    public class FTML : Optimizer
    {
        public FTML(float beta1 = 0.6f, float beta2 = 0.999f, float epsilon = 1e-8f, bool use_fused_step = true) : base(use_fused_step: use_fused_step)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
        }

        public float Beta1 { get; set; }

        public float Beta2 { get; set; }

        public float Epsilon { get; set; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict();
            state["prev_d"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
            state["prev_v"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
            state["prev_z"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
            return state;
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            this.UpdateCount(index);
            var lr = this.GetLr(index);
            var wd = this.GetWd(index);
            var t = this.index_update_count[index];
            // preprocess grad
            grad *= this.RescaleGrad;
            if (this.ClipGradient != null)
            {
                grad = nd.Clip(grad, -this.ClipGradient.Value, this.ClipGradient.Value);
            }

            grad += wd * weight;
            var coef1 = 1.0 - Math.Pow(this.Beta1, t);
            var coef2 = 1.0 - Math.Pow(this.Beta2, t);
            // update d, v, z
            state["prev_v"] *= this.Beta2;
            state["prev_v"] += (1.0 - this.Beta2) * np.square(grad);
            var sigma = -this.Beta1 * state["prev_d"];
            state["prev_d"] = nd.Sqrt(state["prev_v"] / coef2) + this.Epsilon;
            state["prev_d"] *= coef1 / lr;
            sigma += state["prev_d"];
            state["prev_z"] *= this.Beta1;
            state["prev_z"] += (1.0 - this.Beta1) * grad;
            state["prev_z"] -= sigma * weight;
            // update weight
            weight = np.negative(state["prev_z"]) / state["prev_d"];
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);
            var t = index_update_count[index];
            weight = nd.FtmlUpdate(weight, grad, state["prev_d"], state["prev_v"], state["prev_z"], lr, t, Beta1, Beta2,
                Epsilon, wd, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1);
        }
    }
}