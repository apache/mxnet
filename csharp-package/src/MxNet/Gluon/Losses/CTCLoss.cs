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
using System.Linq;

namespace MxNet.Gluon.Losses
{
    public class CTCLoss : Loss
    {
        public CTCLoss(string layout = "NTC", string label_layout = "NT", float? weight = null, int? batch_axis = 0,
            string prefix = "", ParameterDict @params = null) : base(weight, batch_axis)
        {
            if (layout != "NTC" && layout != "TNC")
                throw new ArgumentException($"Only 'NTC' and 'TNC' layouts for pred are supported. Got: {layout}");

            if (label_layout != "NT" && label_layout != "TN")
                throw new ArgumentException(
                    $"Only 'NTC' and 'TNC' layouts for label are supported. Got: {label_layout}");

            Layout = layout;
            LabelLayout = label_layout;
            BatchAxis = label_layout.ToCharArray().ToList().IndexOf('N');
        }

        public string Layout { get; set; }

        public string LabelLayout { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol pred, NDArrayOrSymbol label,
            NDArrayOrSymbol sample_weight = null, params object[] args)
        {
            NDArrayOrSymbol pred_lengths = args.Length > 0 ? (ndarray) args[0] : null;
            NDArrayOrSymbol label_lengths = args.Length > 1 ? (ndarray) args[1] : null;
            if (pred.IsNDArray)
                return F(pred.NdX, label, pred_lengths, label_lengths, sample_weight);

            return F(pred.SymX, label, pred_lengths, label_lengths, sample_weight);
        }

        private ndarray F(ndarray pred, ndarray label, ndarray pred_lengths = null, ndarray label_lengths = null,
            ndarray sample_weight = null)
        {
            if (Layout == "NTC")
                pred = nd.SwapAxis(pred, 0, 1);

            if (BatchAxis == 1)
                label = nd.SwapAxis(label, 0, 1);

            var loss = nd.CTCLoss(pred, label, pred_lengths, label_lengths,
                pred_lengths != null, label_lengths != null,
                CtclossBlankLabel.Last);

            return ApplyWeighting(loss, Weight, sample_weight);
        }

        private _Symbol F(_Symbol pred, _Symbol label, _Symbol pred_lengths = null, _Symbol label_lengths = null,
            _Symbol sample_weight = null)
        {
            if (Layout == "NTC")
                pred = sym.SwapAxis(pred, 0, 1);

            if (BatchAxis == 1)
                label = sym.SwapAxis(label, 0, 1);

            var loss = sym.CTCLoss(pred, label, pred_lengths, label_lengths,
                pred_lengths != null, label_lengths != null,
                CtclossBlankLabel.Last);

            return ApplyWeighting(loss, Weight, sample_weight);
        }
    }
}