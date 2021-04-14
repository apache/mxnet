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
    public class Adamax : Optimizer
    {
        public Adamax(float learning_rate = 0.00f, float beta1 = 0.9f, float beta2 = 0.999f, bool use_fused_step = false) : base(
            learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Beta1 = beta1;
            Beta2 = beta2;
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

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict("mean", "variance");
            state["mean"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            state["variance"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            return state;
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            var t = index_update_count[index];
            lr /= 1 - (float)Math.Pow(Beta1, t);
            grad = grad * RescaleGrad + wd * weight;
            if (ClipGradient.HasValue)
                grad = nd.Clip(grad, -ClipGradient.Value, ClipGradient.Value);

            state["mean"] *= Beta1;
            state["mean"] += (1 - Beta1) * grad;
            state["variance"] = nd.Maximum(Beta2 * state["variance"], nd.Abs(grad));

            weight -= lr * state["mean"] / state["variance"];
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotSupportedException();
        }
    }
}