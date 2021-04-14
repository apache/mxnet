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

namespace MxNet.Gluon.Losses
{
    public class SigmoidBinaryCrossEntropyLoss : Loss
    {
        private readonly bool _from_sigmoid;

        public SigmoidBinaryCrossEntropyLoss(bool from_sigmoid = false, float? weight = null, int? batch_axis = 0,
            string prefix = "", ParameterDict @params = null) : base(weight, batch_axis)
        {
            _from_sigmoid = from_sigmoid;
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            if (label.IsNDArray)
                label = nd.ReshapeLike(label, pred);
            else
                label = sym.ReshapeLike(label, pred);

            NDArrayOrSymbol pos_weight = null;
            NDArrayOrSymbol loss = null;

            if (args.Length > 0)
                pos_weight = args[0] is ndarray
                    ? new NDArrayOrSymbol((ndarray) args[0])
                    : new NDArrayOrSymbol((_Symbol) args[0]);

            if (!_from_sigmoid)
            {
               
                if (pos_weight == null)
                {
                    loss = F.relu(pred) - pred * label +
                              F.activation(F.negative(F.abs(pred)), "softrelu");
                }
                else
                {
                    var log_weight = 1 + F.multiply(pos_weight.NdX - 1, label);
                    loss = F.relu(pred) - pred * label + log_weight
                                                                + F.activation(F.negative(F.abs(pred)), "softrelu")
                                                                + F.relu(F.negative(pred));
                }
            }
            else
            {
                var eps = 1e-12f;
                if (pos_weight == null)
                {
                    loss = F.negative(F.log(pred + eps) * label
                                           + F.log(1 - pred + eps) * (1 - label));
                }
                else
                {
                    loss = F.negative(F.multiply(F.log(pred + eps) * label, pos_weight)
                                           + F.log(1 - pred + eps) * (1 - label));
                }
            }

            loss = ApplyWeighting(loss, Weight, sample_weight);
            if (loss.IsNDArray)
                return nd.Mean(loss, BatchAxis.Value, exclude: true);

            return sym.Mean(loss, BatchAxis.Value, exclude: true);
        }
    }
}