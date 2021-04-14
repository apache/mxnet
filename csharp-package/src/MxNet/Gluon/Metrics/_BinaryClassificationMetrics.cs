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
    internal class BinaryClassificationMetrics : Base
    {
        public int false_negatives;
        public int false_positives;
        public int global_false_negatives;
        public int global_false_positives;
        public int global_true_negatives;

        public int global_true_positives;
        public int true_negatives;
        public int true_positives;

        public float Precision
        {
            get
            {
                if (true_positives + false_positives > 0)
                    return true_positives / (true_positives + (float) false_positives);
                return 0;
            }
        }

        public float GlobalPrecision
        {
            get
            {
                if (global_true_positives + global_false_positives > 0)
                    return global_true_positives / (global_true_positives + (float) global_false_positives);
                return 0;
            }
        }

        public float Recall
        {
            get
            {
                if (true_positives + false_negatives > 0)
                    return true_positives / (true_positives + (float) false_negatives);
                return 0;
            }
        }

        public float GlobalRecall
        {
            get
            {
                if (global_true_positives + global_false_negatives > 0)
                    return global_true_positives / (global_true_positives + (float) global_false_negatives);
                return 0;
            }
        }

        public float FScore
        {
            get
            {
                if (Precision + Recall > 0)
                    return 2f * Precision * Recall / (Precision + Recall);
                return 0;
            }
        }

        public float GlobalFScore
        {
            get
            {
                if (GlobalPrecision + GlobalRecall > 0)
                    return 2f * GlobalPrecision * GlobalRecall / (GlobalPrecision + GlobalRecall);
                return 0;
            }
        }

        public int TotalExamples => false_negatives + false_positives + true_negatives + true_positives;

        public int GlobalTotalExamples => global_false_negatives + global_false_positives + global_true_negatives +
                                          global_true_positives;

        public void UpdateBinaryStats(ndarray label, ndarray pred)
        {
            var pred_label = nd.Argmax(pred, 1);
            CheckLabelShapes(label, pred);
            //ToDo: check unique values and throw error for binary classification

            var pred_true = nd.EqualScalar(pred_label, 1);
            var pred_false = 1 - pred_true;
            var label_true = nd.EqualScalar(label, 1);
            var label_false = 1 - label_true;

            var true_pos = (pred_true * label_true).Sum();
            var false_pos = (pred_true * label_false).Sum();
            var false_neg = (pred_false * label_true).Sum();
            var true_neg = (pred_false * label_false).Sum();

            true_positives += (int) true_pos;
            global_true_positives += (int) true_pos;
            false_positives += (int) false_pos;
            global_false_positives += (int) false_pos;
            false_negatives += (int) false_neg;
            global_false_negatives += (int) false_neg;
            true_negatives += (int) true_neg;
            global_true_negatives += (int) true_neg;
        }

        public float MatthewsCC(bool use_global = false)
        {
            float true_pos, false_pos, false_neg, true_neg;

            if (use_global)
            {
                if (GlobalTotalExamples == 0)
                    return 0;

                true_pos = global_true_positives;
                false_pos = global_false_positives;
                false_neg = global_false_negatives;
                true_neg = global_true_negatives;
            }
            else
            {
                if (TotalExamples == 0)
                    return 0;

                true_pos = true_positives;
                false_pos = false_positives;
                false_neg = false_negatives;
                true_neg = true_negatives;
            }

            var terms = new[]
            {
                true_pos + false_pos,
                true_pos + false_neg,
                true_neg + false_pos,
                true_neg + false_neg
            };

            float denom = 1;

            foreach (var item in terms)
            {
                if (item == 0)
                    continue;

                denom *= item;
            }

            return (true_pos * true_neg - false_pos * false_neg) / (float) Math.Sqrt(denom);
        }

        public void LocalResetStats()
        {
            false_positives = 0;
            false_negatives = 0;
            true_positives = 0;
            true_negatives = 0;
        }

        public void ResetStats()
        {
            false_positives = 0;
            false_negatives = 0;
            true_positives = 0;
            true_negatives = 0;
            global_false_positives = 0;
            global_false_negatives = 0;
            global_true_positives = 0;
            global_true_negatives = 0;
        }
    }
}