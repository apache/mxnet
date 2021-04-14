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
    public class RMSProp : Optimizer
    {
        public RMSProp(float learning_rate = 0.001f, float rho = 0.9f, float momentum = 0.9f,
            float epsilon = 1e-8f, bool centered = false, float? clip_weights = null, bool use_fused_step = true) : base(
            learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Rho = rho;
            Momentum = momentum;
            Epsilon = epsilon;
            Centered = centered;
            ClipWeights = clip_weights.HasValue ? clip_weights.Value : -1;
        }

        public float Rho { get; }
        public float Momentum { get; }
        public float Epsilon { get; }
        public bool Centered { get; }
        public float ClipWeights { get; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict("n", "g", "delta");
            state["mean"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            state["var"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            state["mom"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype).ToSType(weight.stype);
            return state;
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
            if (!this.Centered)
            {
                // update var
                state["var"] *= this.Rho;
                state["var"] += (1 - this.Rho) * np.square(grad);
                // update weight
                var d = grad / (np.sqrt(state["var"]) + this.Epsilon);
                weight -= lr * d;
            }
            else
            {
                // update mean, var, mom
                var _tup_2 = state;
                state["mean"] *= this.Rho;
                state["mean"] += (1 - this.Rho) * grad;
                state["var"] *= this.Rho;
                state["var"] += (1 - this.Rho) * np.square(grad);
                state["mom"] *= this.Momentum;
                state["mom"] -= lr * grad / np.sqrt(state["var"] - np.square(state["mean"]) + this.Epsilon);
                // update weight
                weight[":"] += state["mom"];
            }

            if (this.ClipWeights != 0)
            {
                weight = nd.Clip(weight, -this.ClipWeights, this.ClipWeights);
            }
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);

            if (!Centered)
                weight = nd.RmspropUpdate(weight, grad, state["var"], lr, Rho, Epsilon, wd, RescaleGrad,
                    ClipGradient.HasValue ? ClipGradient.Value : -1, ClipWeights);
            else
                weight = nd.RmspropalexUpdate(weight, grad, state["mean"], state["var"], state["mom"], lr, Rho, Momentum,
                    Epsilon, wd, RescaleGrad, ClipGradient.HasValue ? ClipGradient.Value : -1, ClipWeights);
        }
    }
}