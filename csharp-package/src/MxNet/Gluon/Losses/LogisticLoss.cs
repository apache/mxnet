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
    public class LogisticLoss : Loss
    {
        public LogisticLoss(float? weight = null, int? batch_axis = 0, string label_format = "signed",
            string prefix = "", ParameterDict @params = null) : base(weight, batch_axis)
        {
            if (label_format != "signed" && label_format != "binary")
                throw new ArgumentException($"Label_format can only be signed or binary, recieved {label_format}");

            LabelFormat = label_format;
        }

        public string LabelFormat { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            label = F.reshape_like(label, pred);
            if (LabelFormat == "signed")
                label = (label + 1) / 2;

            var loss = F.relu(pred) - pred * label +
                       F.activation(F.negative(F.abs(pred)), "softrelu");
            loss = ApplyWeighting(loss, Weight, sample_weight);
            return F.mean(loss, BatchAxis.Value);
        }

       
    }
}