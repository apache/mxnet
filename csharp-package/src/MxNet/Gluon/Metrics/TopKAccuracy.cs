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
    public class TopKAccuracy : EvalMetric
    {
        public TopKAccuracy(int top_k = 1, string output_name = null, string label_name = null) : base("top_k_accuracy",
            output_name, label_name, true)
        {
            TopK = top_k;
            if (top_k <= 1)
                throw new ArgumentException("Please use Accuracy if top_k is no more than 1");

            Name = Name + "_" + top_k;
        }

        public int TopK { get; set; }

        public override void Update(ndarray labels, ndarray preds)
        {
            CheckLabelShapes(labels, preds);
            var pred_label = preds.argsort().AsType(DType.Int32); //ToDo: Use numpy argpartition
            var label = labels.AsType(DType.Int32);
            var num_samples = pred_label.shape[0];
            var num_dims = pred_label.shape.Dimension;
            if (num_dims == 1)
            {
                sum_metric += np.equal(pred_label.Ravel(), label.Ravel()).sum().AsScalar<float>();
            }

            else if (num_dims == 2)
            {
                var num_classes = pred_label.shape[1];
                TopK = Math.Min(num_classes, TopK);
                for (var j = 0; j < TopK; j++)
                {
                    float num_correct = nd.Equal(pred_label[$":,{num_classes - 1 - j}"].Ravel(), label.Ravel()).Sum();
                    sum_metric += num_correct;
                    global_sum_metric += num_correct;
                }
            }

            num_inst += num_samples;
            global_num_inst += num_samples;
        }
    }
}