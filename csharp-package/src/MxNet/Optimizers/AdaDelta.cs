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
    public class AdaDelta : Optimizer
    {
        public AdaDelta(float lr = 1, float rho = 0.95f, float decayRate = 0, float epsilon = 1e-07f, bool use_fused_step = false)
            : base(use_fused_step: use_fused_step)
        {
            LearningRate = lr;
            Rho = rho;
            Epsilon = epsilon;
        }

        /// <summary>
        ///     Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
        /// </summary>
        /// <value>
        ///     The rho.
        /// </value>
        public float Rho { get; set; }

        public float Epsilon { get; set; }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            var wd = GetWd(index);
            UpdateCount(index);
            grad *= RescaleGrad;
            if (ClipGradient.HasValue) grad = nd.Clip(grad, -ClipGradient.Value, ClipGradient.Value);

            var acc_g = state["acc_g"];
            var acc_delta = state["acc_delta"];

            acc_g *= Rho;
            acc_g += (1 - Rho) * grad * grad;
            var current_delta = np.sqrt(acc_delta + Epsilon) / np.sqrt(acc_g + Epsilon) * grad;
            acc_delta *= Rho;
            acc_delta += (1 - Rho) * current_delta * current_delta;
            weight -= current_delta + wd * weight;
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotSupportedException();
        }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            return new NDArrayDict
            {
                {"acc_g", nd.Zeros(weight.shape, weight.ctx)},
                {"acc_delta", nd.Zeros(weight.shape, weight.ctx)}
            };
        }
    }
}