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
    public class Speedometer : IBatchEndCallback
    {
        private readonly bool _auto_reset;
        private readonly int _batch_size;
        private readonly int _frequent;
        private bool init;
        private int last_count;
        private long tic;

        public Speedometer(int batch_size, int frequent = 50, bool auto_reset = true)
        {
            _batch_size = batch_size;
            _frequent = frequent;
            _auto_reset = auto_reset;
            init = false;
            tic = 0;
            last_count = 0;
        }

        public void Invoke(int epoch, int nbatch, EvalMetric eval_metric, FuncArgs locals = null)
        {
            var count = nbatch;
            float speed;
            string msg;

            if (last_count > count)
                init = false;

            last_count = count;

            if (init)
            {
                if (count % _frequent == 0)
                {
                    try
                    {
                        speed = (float) Math.Round(_frequent * (float) _batch_size / (DateTime.Now.Ticks - tic));
                    }
                    catch (DivideByZeroException ex)
                    {
                        speed = float.PositiveInfinity;
                    }

                    if (eval_metric != null)
                    {
                        var name_value = eval_metric.GetNameValue();
                        if (_auto_reset)
                        {
                            eval_metric.Reset();
                            msg = string.Format("Epoch[{0}] Batch [{1}-{2}]\tSpeed: {3} samples/sec", epoch,
                                count - _frequent, count, speed);
                            foreach (var item in name_value) msg += string.Format("\t {0}={1}", item.Key, item.Value);

                            Logger.Log(msg);
                        }
                        else
                        {
                            msg = string.Format("Epoch[{0}] Batch [0-{1}]\tSpeed: {2} samples/sec", epoch, count,
                                speed);
                            foreach (var item in name_value) msg += string.Format("\t {0}={1}", item.Key, item.Value);

                            Logger.Log(msg);
                        }
                    }
                    else
                    {
                        Logger.Log(string.Format("Iter[{0}] Batch [{1}]\tSpeed: {} samples/sec", epoch, _batch_size,
                            speed));
                    }

                    tic = DateTime.Now.Ticks;
                }
                else
                {
                    init = true;
                    tic = DateTime.Now.Ticks;
                }
            }
        }
    }
}