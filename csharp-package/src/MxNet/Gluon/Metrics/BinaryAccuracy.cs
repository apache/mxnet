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
    public class BinaryAccuracy : EvalMetric
    {
        public BinaryAccuracy(float threshold = 0.5f, string output_name = null, string label_name = null) : base("accuracy",
            output_name, label_name, true)
        {
            Threshold = threshold;
        }

        public float Threshold { get; }

        public override void Update(ndarray labels, ndarray preds)
        {
            CheckLabelShapes(labels, preds, true);

            preds = preds.clip(0, 1);
            var label = labels.Ravel();
            preds = preds.Ravel() > Threshold;

            var num_correct = nd.Equal(preds, label).AsType(DType.Float32).Sum();

            sum_metric += num_correct;
            global_sum_metric += num_correct;
            num_inst += preds.shape.Size;
            global_num_inst += preds.shape.Size;
        }
    }
}