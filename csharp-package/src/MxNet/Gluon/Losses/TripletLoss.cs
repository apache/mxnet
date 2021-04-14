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
    public class TripletLoss : Loss
    {
        public TripletLoss(float margin = 1, float? weight = null, int? batch_axis = 0, string prefix = null,
            ParameterDict @params = null) : base(weight, batch_axis)
        {
            Margin = margin;
        }

        public float Margin { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            var negative = (NDArrayOrSymbol) args[0];
            if (pred.IsNDArray)
                return F(pred.NdX, label, negative);

            return F(pred.SymX, label, negative);
        }

        private ndarray F(ndarray pred, ndarray positive, ndarray negative)
        {
            positive = nd.ReshapeLike(positive, pred);
            negative = nd.ReshapeLike(negative, pred);
            var loss = nd.Sum(nd.Square(positive - pred) - nd.Square(negative - pred), BatchAxis.Value, exclude: true);
            loss = nd.Relu(loss + Margin);
            loss = ApplyWeighting(loss, Weight);
            return loss;
        }

        private _Symbol F(_Symbol pred, _Symbol positive, _Symbol negative)
        {
            positive = sym.ReshapeLike(positive, pred);
            negative = sym.ReshapeLike(negative, pred);
            var loss = sym.Sum(sym.Square(positive - pred) - sym.Square(negative - pred), BatchAxis.Value,
                exclude: true);
            loss = sym.Relu(loss + Margin);
            loss = ApplyWeighting(loss, Weight);
            return loss;
        }
    }
}