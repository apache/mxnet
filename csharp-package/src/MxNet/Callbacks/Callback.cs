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
    //ToDo: Removed on v2.0
    //public class ModuleCheckpoint : IIterEndCallback
    //{
    //    private readonly Module _mod;
    //    private int _period;
    //    private readonly string _prefix;
    //    private readonly bool _save_optimizer_states;

    //    public ModuleCheckpoint(Module mod, string prefix, int period = 1, bool save_optimizer_states = false)
    //    {
    //        _mod = mod;
    //        _prefix = prefix;
    //        _period = period;
    //        _save_optimizer_states = save_optimizer_states;
    //    }

    //    public void Invoke(int epoch)
    //    {
    //        _period = Math.Max(1, _period);
    //        if ((epoch + 1) % _period == 0) _mod.SaveCheckpoint(_prefix, epoch + 1, _save_optimizer_states);
    //    }
    //}


    public class DoCheckPoint : IEpochEndCallback
    {
        private int _period;
        private readonly string _prefix;

        public DoCheckPoint(string prefix, int period = 1)
        {
            _prefix = prefix;
            _period = period;
        }

        public void Invoke(int epoch, Symbol symbol, NDArrayDict arg_params, NDArrayDict aux_params)
        {
            _period = Math.Max(1, _period);
            if ((epoch + 1) % _period == 0) MxModel.SaveCheckpoint(_prefix, epoch + 1, symbol, arg_params, aux_params);
        }
    }

    public class LogTrainMetric : IIterEpochCallback
    {
        private readonly bool _auto_reset;
        private readonly int _period;

        public LogTrainMetric(int period, bool auto_reset = false)
        {
            _auto_reset = auto_reset;
            _period = period;
        }

        public void Invoke(int epoch, int nbatch, EvalMetric eval_metric, FuncArgs locals = null)
        {
            if (nbatch % _period == 0 && eval_metric != null)
            {
                var name_values = eval_metric.GetNameValue();
                foreach (var item in name_values)
                    Logger.Log(string.Format("Iter: {0} Batch: {1} Train-{2}={3}", epoch, nbatch, item.Key,
                        item.Value));

                if (_auto_reset)
                    eval_metric.Reset();
            }
        }
    }
}