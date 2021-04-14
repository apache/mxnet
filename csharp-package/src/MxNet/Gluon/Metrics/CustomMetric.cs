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
    public class CustomMetric : EvalMetric
    {
        private bool _allow_extra_outputs;
        private readonly Func<ndarray, ndarray, float> _feval;

        public CustomMetric(Func<ndarray, ndarray, float> feval, string name, string output_name = null,
            string label_name = null, bool has_global_stats = false)
            : base(string.Format("custom({0})", name), output_name, label_name, has_global_stats)
        {
            _feval = feval;
        }

        public override void Update(ndarray labels, ndarray preds)
        {
            CheckLabelShapes(labels, preds);
            var reval = _feval(labels, preds);
            num_inst++;
            global_num_inst++;
            sum_metric += reval;
            global_sum_metric += reval;
        }

        public override ConfigData GetConfig()
        {
            throw new NotImplementedException("Custom metric cannot be serialized");
        }
    }
}