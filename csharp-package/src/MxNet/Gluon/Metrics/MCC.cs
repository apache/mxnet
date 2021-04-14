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

namespace MxNet.Gluon.Metrics
{
    public class MCC : EvalMetric
    {
        private readonly BinaryClassificationMetrics metrics;

        public MCC(string output_name = null, string label_name = null, string average = "macro") : base("mcc",
            output_name, label_name, true)
        {
            Average = average;
            metrics = new BinaryClassificationMetrics();
        }

        public string Average { get; }

        public override void Update(ndarray labels, ndarray preds)
        {
            CheckLabelShapes(labels, preds);

            metrics.UpdateBinaryStats(labels, preds);

            if (Average == "macro")
            {
                sum_metric += metrics.MatthewsCC();
                global_sum_metric += metrics.MatthewsCC(true);
                num_inst += 1;
                global_num_inst += 1;
                metrics.ResetStats();
            }
            else
            {
                sum_metric = metrics.MatthewsCC() * metrics.TotalExamples;
                global_sum_metric = metrics.MatthewsCC(true) * metrics.GlobalTotalExamples;
                num_inst = metrics.TotalExamples;
                global_num_inst = metrics.GlobalTotalExamples;
            }
        }

        public override void Reset()
        {
            base.Reset();
            metrics.ResetStats();
        }

        public override void ResetLocal()
        {
            base.ResetLocal();
            metrics.LocalResetStats();
        }
    }
}