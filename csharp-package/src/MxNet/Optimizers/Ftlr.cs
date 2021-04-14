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
    public class Ftlr : Optimizer
    {
        public Ftlr(float lamda1 = 0.1f, float learning_rate = 0.1f, float beta = 1, bool use_fused_step = true) 
            : base(learning_rate: learning_rate, use_fused_step: use_fused_step)
        {
            Lamda1 = lamda1;
            Beta = beta;
        }

        public float Lamda1 { get; set; }

        public float Beta { get; set; }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            var state = new NDArrayDict();
            state["z"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
            state["n"] = nd.Zeros(weight.shape, weight.ctx, weight.dtype);
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
            // update z, n
            var sigma = np.negative(np.sqrt(state["n"]));
            state["n"] += np.square(grad);
            var denom = np.sqrt(state["n"]);
            sigma += denom;
            sigma /= lr;
            state["z"] += grad - sigma * weight;
            // update weight
            denom += this.Beta;
            denom /= lr;
            denom += wd;
            var d = np.sign(state["z"]) * np.maximum(nd.Abs(state["z"]) - this.Lamda1, 0);
            weight = np.negative(d) / denom;
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);
            var t = index_update_count[index];
            weight = nd.FtrlUpdate(weight, grad, state["z"], state["n"], lr, Lamda1, Beta, wd, RescaleGrad,
                ClipGradient.HasValue ? ClipGradient.Value : -1);
        }
    }
}