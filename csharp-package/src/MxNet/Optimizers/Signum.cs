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

namespace MxNet.Optimizers
{
    public class Signum : Optimizer
    {
        public Signum(float learning_rate = 0.01f, float momentum = 0.9f, float wd_lh = 0, bool use_fused_step = true) : base(
            learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Momentum = momentum;
            WdLh = wd_lh;
        }

        public float Momentum { get; set; }

        public float WdLh { get; set; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict();
            state["momentum"] = null;

            if (Momentum != 0)
                state["momentum"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            return state;
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            this.UpdateCount(index);
            var lr = this.GetLr(index);
            var wd = this.GetWd(index);
            if (state != null)
            {
                // preprocess grad
                grad *= this.RescaleGrad;
                if (this.ClipGradient != null)
                {
                    grad = nd.Clip(grad, -this.ClipGradient.Value, this.ClipGradient.Value);
                }
                grad += wd * weight;
                // update mom
                var mom = state["momentum"];
                mom *= this.Momentum;
                mom -= (1 - this.Momentum) * grad;
                // update weight
                weight *= 1 - lr * this.WdLh;
                weight += lr * ((mom > 0) - (mom < 0));
            }
            else
            {
                // update weight
                weight *= 1 - lr * (wd + this.WdLh);
                weight -= lr * ((grad > 0) - (grad < 0));
            }
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            if (state["momentum"] != null)
                weight = nd.SignumUpdate(weight, grad, state["momentum"], lr, Momentum, wd, RescaleGrad,
                    ClipGradient.HasValue ? ClipGradient.Value : -1, WdLh);
            else
                weight = nd.SignsgdUpdate(weight, grad, lr, wd, RescaleGrad,
                    ClipGradient.HasValue ? ClipGradient.Value : -1);
        }
    }
}