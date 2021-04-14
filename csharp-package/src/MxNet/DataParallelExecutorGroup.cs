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
using System.Collections.Generic;
using System.Linq;
using MxNet.IO;
using MxNet.Metrics;

namespace MxNet
{
    public class DataParallelExecutorGroup
    {
        internal NDArrayList aux_arrays = new NDArrayList();
        internal List<string> aux_names = new List<string>();

        internal NDArrayList data_arrays = new NDArrayList();
        internal List<string> data_names = new List<string>();
        internal NDArrayList grad_arrays = new NDArrayList();
        internal NDArrayList label_arrays = new NDArrayList();
        internal List<string> label_names = new List<string>();
        internal NDArrayList param_arrays = new NDArrayList();
        internal List<int> param_idx = new List<int>();
        internal List<string> param_names = new List<string>();
        internal List<NDArrayDict> shared_data_arrays = new List<NDArrayDict>();

        private readonly Slice[] slices;

        internal List<Executor> train_execs = new List<Executor>();

        public DataParallelExecutorGroup(Symbol sym, string[] arg_names, string[] param_names, Context[] ctxlist,
            Slice[] slices, DataIter train_data, DataParallelExecutorGroup shared_group = null)
        {
            ExecuterManager.CheckArguments(sym);

            if (shared_group == null)
                foreach (var item in ctxlist)
                    shared_data_arrays.Add(new NDArrayDict());
            else
                shared_data_arrays = shared_group.shared_data_arrays;

            foreach (var item in train_data.ProvideData)
                data_names.Add(item.Name);

            foreach (var item in train_data.ProvideLabel)
                label_names.Add(item.Name);

            aux_names = sym.ListAuxiliaryStates().ToList();
            for (var i = 0; i < arg_names.Length; i++)
                if (param_names.Contains(arg_names[i]))
                {
                    param_idx.Add(i);
                    this.param_names.Add(arg_names[i]);
                }

            for (var i = 0; i < ctxlist.Length; i++)
            {
                var data_shapes = new Dictionary<string, Shape>();
                var data_types = new Dictionary<string, DType>();
                var shapeData = new List<int>();
                foreach (var item in train_data.ProvideData)
                {
                    shapeData = item.Shape.Data.ToList();
                    shapeData.RemoveAt(0);
                    shapeData.Insert(0, slices[i].End.Value - slices[i].Begin);
                    data_shapes[item.Name] = new Shape(shapeData);
                    data_types[item.Name] = item.DataType;
                }

                foreach (var item in train_data.ProvideLabel)
                {
                    shapeData = item.Shape.Data.ToList();
                    shapeData.RemoveAt(0);
                    shapeData.Insert(0, slices[i].End.Value - slices[i].Begin);
                    data_shapes[item.Name] = new Shape(shapeData);
                    data_types[item.Name] = item.DataType;
                }

                var shared_exec = shared_group == null ? null : shared_group.train_execs[i];
                var train_exec = ExecuterManager.BindExec(sym, ctxlist[i], data_shapes, param_names, true,
                    shared_exec, shared_data_arrays[i], data_types);

                train_execs.Add(train_exec);
            }

            foreach (var name in data_names)
                for (var i = 0; i < train_execs.Count; i++)
                    data_arrays.Add(train_execs[i].ArgmentDictionary()[name]);

            foreach (var name in label_names)
                for (var i = 0; i < train_execs.Count; i++)
                    label_arrays.Add(train_execs[i].ArgmentDictionary()[name]);

            foreach (var idx in param_idx)
                for (var i = 0; i < train_execs.Count; i++)
                    param_arrays.Add(train_execs[i].ArgmentArrays[idx]);

            foreach (var idx in param_idx)
                for (var i = 0; i < train_execs.Count; i++)
                    grad_arrays.Add(train_execs[i].GradientArrays[idx]);

            for (var idx = 0; idx < aux_names.Count; idx++)
            for (var i = 0; i < train_execs.Count; i++)
                aux_arrays.Add(train_execs[i].AuxiliaryArrays[i]);

            this.slices = slices;
        }

        public void LoadDataBatch(DataBatch data_batch)
        {
            ExecuterManager.LoadData(data_batch, data_arrays.ToArray());
            ExecuterManager.LoadData(data_batch, label_arrays.ToArray());
        }

        public void Forward(bool is_train = false)
        {
            foreach (var exec in train_execs) exec.Forward(is_train);
        }

        public void Backward()
        {
            foreach (var exec in train_execs) exec.Backward();
        }

        public void UpdateMetric(EvalMetric metric, NDArrayList labels, bool pre_sliced = false)
        {
            var labels_slice = new NDArrayList();
            var i = 0;
            train_execs.Zip(slices, (e, s) =>
            {
                if (!pre_sliced)
                    foreach (var label in labels)
                        labels_slice.Add(label.Slice(s.Begin, s.End.Value));
                else
                    labels_slice.Add(labels[i]);

                metric.Update(labels_slice.ToArray(), e.Outputs.ToArray());
                i++;
                return true;
            });
        }
    }
}