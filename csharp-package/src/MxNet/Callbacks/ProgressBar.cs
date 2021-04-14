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
using System;
using MxNet.Gluon.Metrics;

namespace MxNet.Callbacks
{
    public class ProgressBar : IBatchEndCallback
    {
        private readonly int length;
        private readonly int total;

        public ProgressBar(int total, int length = 80)
        {
            this.total = total;
            this.length = length;
        }

        public void Invoke(int epoch, int nbatch, EvalMetric eval_metric, FuncArgs locals = null)
        {
            var count = nbatch;
            var filled_len = Convert.ToInt32(length * count / (float) total);
            var percents = Math.Ceiling(100 * count / (float) total);
            var prog_bar = "";
            for (var i = 0; i < filled_len; i++) prog_bar += "=";

            for (var i = 0; i < length - filled_len; i++) prog_bar += "-";

            Logger.Log(string.Format("[{0}] {1}%", prog_bar, Math.Round(percents)));
        }
    }
}