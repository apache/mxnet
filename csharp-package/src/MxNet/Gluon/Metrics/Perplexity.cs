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

namespace MxNet.Gluon.Metrics
{
    public class Perplexity : EvalMetric
    {
        public Perplexity(int? ignore_label, int axis = -1, string output_name = null, string label_name = null) : base(
            "perplexity", output_name, label_name, true)
        {
            IgnoreLabel = ignore_label;
            Axis = axis;
        }

        public int? IgnoreLabel { get; }

        public int Axis { get; }

        public override void Update(ndarray labels, ndarray preds)
        {
            float loss = 0;
            long num = 0;

            labels = labels.AsInContext(preds.ctx).reshape(Convert.ToInt32(preds.size));
            preds = nd.Pick(preds, labels.AsType(DType.Int32), Axis);
            if (IgnoreLabel.HasValue)
            {
                var ignore = np.equal(labels, IgnoreLabel.Value).AsType(preds.dtype);
                num -= nd.Sum(ignore).AsScalar<int>();
                preds = preds * (1 - ignore) + ignore;
            }

            loss -= nd.Sum(nd.Log(nd.MaximumScalar(preds, 1e-10f))).AsScalar<float>();
            num += preds.size;

            sum_metric += loss;
            global_sum_metric += loss;
            num_inst += num;
            global_num_inst += num;
        }

        public override (string, float) Get()
        {
            if (num_inst == 0)
                return (Name, float.NaN);

            return (Name, (float) Math.Exp(sum_metric / num_inst));
        }

        public override (string, float) GetGlobal()
        {
            if (hasGlobalStats)
            {
                if (global_num_inst == 0)
                    return (Name, float.NaN);

                return (Name, (float) Math.Exp(global_sum_metric / global_num_inst));
            }

            return Get();
        }
    }
}