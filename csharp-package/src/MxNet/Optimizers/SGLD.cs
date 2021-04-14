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
    public class SGLD : Optimizer
    {
        public SGLD(float learning_rate = 0.1f, bool use_fused_step = false) : base(learning_rate: learning_rate, use_fused_step: use_fused_step)
        {

        }

        public override NDArrayDict CreateState(int index, ndarray weight)
        {
            return new NDArrayDict();
        }

        public override void Step(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            UpdateCount(index);
            var lr = GetLr(index);
            var wd = GetWd(index);
            grad = grad * RescaleGrad;
            if (ClipGradient.HasValue)
                grad = nd.Clip(grad, -ClipGradient.Value, ClipGradient.Value);

            weight += -lr / 2 * (grad + wd * weight);
            weight += np.random.normal(0, (float)Math.Sqrt(lr), weight.shape, dtype: weight.dtype,
                ctx: weight.ctx);
        }

        public override void FusedStep(int index, ndarray weight, ndarray grad, NDArrayDict state)
        {
            throw new NotSupportedException();
        }
    }
}