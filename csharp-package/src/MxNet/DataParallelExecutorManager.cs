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
using System.Collections.Generic;
using System.Linq;
using MxNet.IO;
using MxNet.Metrics;

namespace MxNet
{
    public class DataParallelExecutorManager
    {
        internal readonly string[] arg_names;
        internal readonly string[] aux_names;
        internal readonly Context[] contexts;
        internal DataParallelExecutorGroup curr_execgrp;
        internal DataParallelExecutorGroup execgrp;

        internal readonly Dictionary<int, DataParallelExecutorGroup> execgrp_bucket =
            new Dictionary<int, DataParallelExecutorGroup>();

        internal readonly int num_device;
        internal readonly string[] param_names;
        internal readonly Slice[] slices;
        internal readonly Func<int, Symbol> sym_gen;
        internal Symbol symbol;

        public DataParallelExecutorManager(Symbol symbol, Context[] ctx, DataIter train_data, string[] arg_names,
            string[] param_names,
            string[] aux_names, int[] work_load_list = null, Logger logger = null, Func<int, Symbol> sym_gen = null)
        {
            num_device = ctx.Length;
            Logger.Info(string.Format("Start training with {0}", num_device));

            if (work_load_list == null)
            {
                work_load_list = new int[num_device];
                for (var i = 0; i < num_device; i++)
                    work_load_list[i] = 1;
            }
            else if (work_load_list.Length != num_device)
            {
                throw new MXNetException("Invalid setting for work load");
            }

            slices = ExecuterManager.SplitInputSlice(train_data.BatchSize, work_load_list);

            this.arg_names = arg_names;
            this.param_names = param_names;
            this.aux_names = aux_names;
            contexts = ctx;
            execgrp = new DataParallelExecutorGroup(symbol, arg_names, param_names, ctx, slices, train_data);
            this.symbol = symbol;
            this.sym_gen = sym_gen;
            if (sym_gen != null)
                execgrp_bucket.Add(train_data.DefaultBucketKey, execgrp);
        }

        public NDArrayList ParamArrays => execgrp.param_arrays.ToArray();

        public NDArrayList GradArrays => execgrp.grad_arrays.ToArray();

        public NDArrayList AuxArrays => execgrp.aux_arrays.ToArray();

        public void InstallMonitor(Monitor monitor)
        {
            if (sym_gen != null)
                throw new MXNetException("Monitoring is not implemented for bucketing");

            foreach (var texec in execgrp.train_execs) monitor.Install(texec);
        }

        public void SetParams(NDArrayDict arg_params, NDArrayDict aux_params)
        {
            foreach (var texec in execgrp.train_execs) texec.CopyFromParams(arg_params, aux_params);
        }

        public void CopyTo(NDArrayDict arg_params, NDArrayDict aux_params)
        {
            //ToDo: Revisit code
            param_names.Zip(ParamArrays, (name, block) =>
            {
                var w = new NDArray(new[] {block.Sum()}, Context.Cpu());
                w.AsType(arg_params[name].DataType).CopyTo(arg_params[name]);
                return true;
            });

            aux_names.Zip(AuxArrays, (name, block) =>
            {
                var w = new NDArray(new[] {block.Sum()}, Context.Cpu());
                w.AsType(aux_params[name].DataType).CopyTo(aux_params[name]);
                return true;
            });
        }

        public void LoadDataBatch(DataBatch data_batch)
        {
            if (sym_gen != null)
            {
                var key = data_batch.BucketKey.Value;
                if (execgrp_bucket.ContainsKey(key))
                {
                    symbol = sym_gen(key);
                    execgrp = new DataParallelExecutorGroup(symbol, arg_names, param_names, contexts, slices,
                        NDArrayIter.FromBatch(data_batch), execgrp);
                    execgrp_bucket[key] = execgrp;
                }

                curr_execgrp = execgrp_bucket[key];
            }
            else
            {
                curr_execgrp = execgrp;
            }

            curr_execgrp.LoadDataBatch(data_batch);
        }

        public void Forward(bool is_train = false)
        {
            curr_execgrp.Forward(is_train);
        }

        public void Backward()
        {
            curr_execgrp.Backward();
        }

        public void UpdateMetric(EvalMetric eval_metric, NDArrayList labels, bool pre_sliced = false)
        {
            curr_execgrp.UpdateMetric(eval_metric, labels, pre_sliced);
        }
    }
}