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
using MxNet.Sym.Numpy;
using System;

namespace MxNet.Gluon.Losses
{
    public class PoissonNLLLoss : Loss
    {
        public PoissonNLLLoss(bool from_logits = false, bool compute_full = false, float? weight = null,
            int? batch_axis = 0, string prefix = "", ParameterDict @params = null) : base(weight, batch_axis, prefix,
            @params)
        {
            FromLogit = from_logits;
            ComputeFull = compute_full;
        }

        public bool FromLogit { get; set; }

        public bool ComputeFull { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            label = F.reshape_like(label, pred);
            NDArrayOrSymbol loss = null;
            if (FromLogit)
                loss = F.exp(pred) - label * pred;
            else
                loss = pred - label * F.log(pred + 1e-08f);

            if (ComputeFull)
            {
                var stirling_factor = label * F.log(label) - label + 0.5f * F.log(2 * label * (float)Math.PI);
                var target_gt_1 = label > 1;
                stirling_factor *= target_gt_1;
                loss += stirling_factor;
            }

            loss = ApplyWeighting(loss, Weight, sample_weight);
            return F.mean(loss);
        }
    }
}