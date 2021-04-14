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
    public class NAG : Optimizer
    {
        public NAG(float learning_rate = 0.1f, float momentum = 0, bool multi_precision = false, bool use_fused_step = false)
            : base(learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Momentum = momentum;
            MultiPrecision = multi_precision;
        }

        public float Momentum { get; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict("momentum");
            if (Momentum != 0)
                state["momentum"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);

            return state;
        }

        public override (NDArrayDict, ndarray) CreateStateMultiPrecision(int index, ndarray weight)
        {
            return base.CreateStateMultiPrecision(index, weight);
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
            ndarray d;
            // update mom
            if (state["momentum"] != null)
            {
                state["momentum"] *= this.Momentum;
                state["momentum"] -= lr * grad;
                d = this.Momentum * state["momentum"] - lr * grad;
            }
            else
            {
                d = -lr * grad;
            }

            // update weight
            weight += d;
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            _update_impl(index, weight, grad, state);
        }

        private void _update_impl(int index, ndarray weight, ndarray grad, NDArrayDict state,
            bool multi_precision = false)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            if (!multi_precision)
            {
                if (state["momentum"] != null)
                    weight = nd.NAGMomUpdate(weight, grad, state["momentum"], lr, Momentum, wd, RescaleGrad,
                        ClipGradient.HasValue ? ClipGradient.Value : -1);
                else
                    weight = nd.SgdUpdate(weight, grad, lr, wd, RescaleGrad,
                        ClipGradient.HasValue ? ClipGradient.Value : -1);
            }
            else
            {
                if (state["momentum"] != null)
                    weight = nd.MPNAGMomUpdate(weight, grad, state["momentum"], state["weight32"], lr, Momentum, wd,
                        RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1);
                else
                    weight = nd.MpSgdUpdate(weight, grad, state["weight32"], lr, wd, RescaleGrad,
                        ClipGradient.HasValue ? ClipGradient.Value : -1);
            }
        }

        public override void UpdateMultiPrecision(int index, ndarray weight, ndarray grad, (NDArrayDict, ndarray) state)
        {
            base.UpdateMultiPrecision(index, weight, grad, state);
        }
    }
}