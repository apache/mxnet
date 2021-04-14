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
using System.Linq;

namespace MxNet.Gluon.Losses
{
    public class SoftmaxCrossEntropyLoss : Loss
    {
        private readonly int _axis;
        private readonly bool _from_logits;
        private readonly bool _sparse_label;

        public SoftmaxCrossEntropyLoss(int axis = -1, bool sparse_label = true, bool from_logits = false,
            float? weight = null, int? batch_axis = 0, string prefix = "", ParameterDict @params = null) : base(
            weight, batch_axis)
        {
            _axis = axis;
            _sparse_label = sparse_label;
            _from_logits = from_logits;
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            if (pred.IsNDArray)
                return F(pred.NdX, label.NdX, sample_weight != null ? sample_weight.NdX : null);

            return F(pred.SymX, label.SymX, sample_weight != null ? sample_weight.SymX : null);
        }

        private ndarray F(ndarray pred, ndarray label, ndarray sample_weight = null)
        {
            ndarray loss = null;
            if (!_from_logits)
                pred = nd.LogSoftmax(pred, _axis);

            if (_sparse_label)
            {
                loss = nd.Negative(nd.Pick(pred, label, _axis, true));
            }
            else
            {
                label = nd.ReshapeLike(label, pred);
                loss = nd.Negative(nd.Sum(pred * label, _axis, true));
            }

            loss = ApplyWeighting(loss, Weight, sample_weight).NdX;
            return nd.Mean(loss, BatchAxis.Value, exclude: true);
        }

        private _Symbol F(_Symbol pred, _Symbol label, _Symbol sample_weight = null)
        {
            _Symbol loss = null;
            if (_from_logits)
                pred = sym.LogSoftmax(pred, _axis);

            if (_sparse_label)
            {
                loss = sym.Negative(sym.Pick(pred, label, _axis, true));
            }
            else
            {
                label = sym.ReshapeLike(label, pred);
                loss = sym.Negative(sym.Sum(pred * label, _axis, true));
            }

            

            loss = ApplyWeighting(loss, Weight, sample_weight).SymX;
            return sym.Mean(loss, BatchAxis.Value, exclude: true);
        }
    }

    public class SoftmaxCELoss : SoftmaxCrossEntropyLoss
    {
        public SoftmaxCELoss(int axis = -1, bool sparse_label = true, bool from_logits = false,
            float? weight = null, int? batch_axis = 0, string prefix = "", ParameterDict @params = null) : base(
            axis, sparse_label, from_logits, weight, batch_axis)
        {
        }
    }
}