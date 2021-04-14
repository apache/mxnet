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
    public class Adam : Optimizer
    {
        public Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
            bool lazy_update = true, bool use_fused_step = true) : base(learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
            LazyUpdate = lazy_update;
        }


        /// <summary>
        ///     Gets or sets the beta 1 value.
        /// </summary>
        /// <value>
        ///     The beta1.
        /// </value>
        public float Beta1 { get; set; }

        /// <summary>
        ///     Gets or sets the beta 2 value.
        /// </summary>
        /// <value>
        ///     The beta2.
        /// </value>
        public float Beta2 { get; set; }

        public float Epsilon { get; set; }

        public bool LazyUpdate { get; set; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var stype = LazyUpdate ? weight.stype : StorageStype.Default;
            var state = new NDArrayDict("mean", "variance");
            state["mean"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(stype);
            state["variance"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(stype);
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
            lr *= Convert.ToSingle(Math.Sqrt(coef2) / coef1);
            // update mean and var
            state["mean"] *= this.Beta1;
            state["mean"] += (1.0 - this.Beta1) * grad;
            state["variance"] *= this.Beta2;
            state["variance"] += (1.0 - this.Beta2) * np.square(grad);
            // update weight
            var d = state["mean"] / (np.sqrt(state["variance"]) + this.Epsilon);
            weight -= lr * d;
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            var t = index_update_count[index];

            var coef1 = 1 - (float) Math.Pow(Beta1, t);
            var coef2 = 1 - (float) Math.Pow(Beta2, t);

            lr *= (float) Math.Sqrt(coef2) / coef1;
            weight = nd.AdamUpdate(weight, grad, state["mean"], state["variance"], lr, Beta1, Beta2, Epsilon, wd,
                RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1, LazyUpdate);
        }
    }
}