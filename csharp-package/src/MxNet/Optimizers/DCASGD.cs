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
    public class DCASGD : Optimizer
    {
        public DCASGD(float momentum = 0, float lamda = 0.04f, bool use_fused_step = false) : base (use_fused_step: use_fused_step)
        {
            Momentum = momentum;
            Lamda = lamda;
        }

        public float Momentum { get; set; }

        public float Lamda { get; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict("momentum", "prev_weight");
            if (Momentum == 0)
            {
                state["momentum"] = null;
                state["prev_weight"] = weight.Copy();
            }
            else
            {
                state["momentum"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
                state["prev_weight"] = weight.Copy();
            }

            return state;
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            grad = grad * RescaleGrad;
            if (ClipGradient.HasValue)
                grad = nd.Clip(grad, -ClipGradient.Value, ClipGradient.Value);

            if (state["momentum"] != null)
            {
                state["momentum"] *= Momentum;
                state["momentum"] += -lr * (grad + wd * weight + Lamda * grad * grad * (weight - state["prev_weight"]));
            }
            else
            {
                state["momentum"] += -lr * (grad + wd * weight + Lamda * grad * grad * (weight - state["prev_weight"]));
            }

            state["prev_weight"] = weight;
            weight += state["momentum"];
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotSupportedException();
        }
    }
}