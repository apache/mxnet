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
using MxNet.IO;
using MxNet.Metrics;
using System.Linq;

namespace MxNet.Modules
{
    public class DataParallelExecutorGroup
    {
        public NDArrayDict ArgParams { get; set; }
        public NDArrayDict AuxParams { get; set; }
        public Symbol Symbol { get; }
        public Context[] Contexts { get; }
        public int[] Workload { get; }
        
        public bool ForTraining { get; }
        public bool InputsNeedGrad { get; }
        public string[] ParamNames { get; set; }
        public string[] DataNames { get; set; }
        public string[] LabelNames { get; set; }
        public string[] FixedParamNames { get; }
        public string[] StateNames { get; }
        public string[] OutputNames { get; }
        public Dictionary<string, Context>[]Group2Ctxs { get; }
        public Dictionary<string, OpGradReq> GradReq { get; }
        public string[] ArgNames { get; set; }
        public string[] AuxNames { get; set; }
        public List<List<(Slice, NDArray)>> DataArrays { get; set; }
        public List<List<(Slice, NDArray)>> LabelArrays { get; set; }
        public List<NDArrayList> StateArrays { get; set; }
        public List<NDArrayList> ParamArrays { get; set; }
        public List<NDArrayList> GradArrays { get; set; }
        public List<NDArrayList> InputGradArrays { get; set; }
        public List<NDArrayList> AuxArrays { get; set; }
        public int? BatchSize { get; set; }
        public List<Executor> Execs { get; set; }
        public List<NDArrayDict> SharedDataArrays { get; set; }
        public Slice[] Slices { get; set; }

        public DataDesc[] DataShapes { get; set; }
        public DataDesc[] LabelShapes { get; set; }
        public int[] DataLayouts { get; set; }
        public int[] LabelLayouts { get; set; }
        public int[] OutputLayouts { get; set; }
        public int NumOutputs { get; set; }

        internal int _total_exec_bytes;
        internal Executor[] _default_execs;

        public DataParallelExecutorGroup(Symbol symbol, Context[] contexts, int[] workload, DataDesc[] data_shapes,
            DataDesc[] label_shapes,
            string[] param_names, bool for_training, bool inputs_need_grad,
            DataParallelExecutorGroup shared_group = null,
            string[] fixed_param_names = null, OpGradReq grad_req = OpGradReq.Write, string[] state_names = null,
            Dictionary<string, Context>[] group2ctxs = null)
        {
            Symbol = symbol;
            Contexts = contexts;
            Workload = workload;
            ParamNames = param_names;
            ForTraining = for_training;
            InputsNeedGrad = inputs_need_grad;
            FixedParamNames = fixed_param_names;
            StateNames = state_names;
            Group2Ctxs = ExecutorGroup.PrepareGroup2Ctxs(group2ctxs, contexts.Length);
            ArgNames = symbol.ListArguments().ToArray();
            AuxNames = symbol.ListAuxiliaryStates().ToArray();
            Execs = new List<Executor>();
            _total_exec_bytes = 0;

            if (FixedParamNames == null)
                FixedParamNames = new string[0];

            if (StateNames == null)
                StateNames = new string[0];

            if (!ForTraining)
                grad_req = OpGradReq.Null;

            var data_names = data_shapes.Select(x => x.Name).ToArray();
            GradReq = new Dictionary<string, OpGradReq>();
            foreach (var name in ArgNames)
            {
                GradReq.Add(name, grad_req);
            }

            if (shared_group != null)
                SharedDataArrays = shared_group.SharedDataArrays;
            else
            {
                SharedDataArrays = new List<NDArrayDict>();
                for (int i = 0; i < contexts.Length; i++)
                {
                    SharedDataArrays.Add(new NDArrayDict());
                }
            }

            OutputNames = symbol.ListOutputs().ToArray();
            OutputLayouts = OutputNames.Select(x=>(DataDesc.GetBatchAxis(symbol[x].Attr("__layout__")))).ToArray();
            NumOutputs = symbol.ListOutputs().Count;
            BindExec(data_shapes, label_shapes, shared_group);
        }

        public int[] DecideSlices(DataDesc[] data_shapes)
        {
            if (data_shapes.Length == 0)
                throw new ArgumentNullException("data_shapes", "null ot 0 element");

            var major_axis = data_shapes.Select(x => (DataDesc.GetBatchAxis(x.Layout))).ToArray();
            for (int i = 0; i < data_shapes.Length; i++)
            {
                int axis = major_axis[i];
                var ds = data_shapes[i];
                if (axis != -1)
                {
                    int batch_size = ds.Shape[axis];
                    if (BatchSize.HasValue && batch_size != BatchSize.Value)
                        throw new Exception($"all data must have the same batch size: batch_size = {BatchSize.Value}, but {ds.Name} has shape {ds.Shape}");
                    else
                    {
                        BatchSize = batch_size;
                        Slices = ExecuterManager.SplitInputSlice(batch_size, Workload);
                    }

                }
            }

            return major_axis;
        }

        public void CollectArrays()
        {
            DataArrays = new List<List<(Slice, NDArray)>>();
            LabelArrays = new List<List<(Slice, NDArray)>>();
            foreach (var ds in DataShapes)
            {
                List<(Slice, NDArray)> arrays = new List<(Slice, NDArray)>();
                for (int i = 0; i < Execs.Count; i++)
                {
                    var e = Execs[i];
                    var arg_dict = e.ArgmentDictionary();
                    arrays.Add((Slices[i], arg_dict[ds.Name]));
                }

                DataArrays.Add(arrays);
            }

            if (LabelShapes != null)
            {
                foreach (var ls in LabelShapes)
                {
                    List<(Slice, NDArray)> arrays = new List<(Slice, NDArray)>();
                    for (int i = 0; i < Execs.Count; i++)
                    {
                        var e = Execs[i];
                        var arg_dict = e.ArgmentDictionary();
                        arrays.Add((Slices[i], arg_dict[ls.Name]));
                    }

                    LabelArrays.Add(arrays);
                }
            }

            StateArrays = new List<NDArrayList>();
            foreach (var name in StateNames)
            {
                NDArrayList arrays = new NDArrayList();
                for (int i = 0; i < Execs.Count; i++)
                {
                    var e = Execs[i];
                    var arg_dict = e.ArgmentDictionary();
                    arrays.Add(arg_dict[name]);
                }

                StateArrays.Add(arrays);
            }

            ParamArrays = new List<NDArrayList>();
            for (int i = 0; i < ArgNames.Length; i++)
            {
                var name = ArgNames[i];

                if (!ParamNames.Contains(name))
                    continue;

                NDArrayList arrays = new NDArrayList();
                foreach (var e in Execs)
                {
                    arrays.Add(e.ArgmentArrays[i]);
                }

                ParamArrays.Add(arrays);
            }

            GradArrays = new List<NDArrayList>();
            if(ForTraining)
            {
                for (int i = 0; i < ArgNames.Length; i++)
                {
                    var name = ArgNames[i];

                    if (!ParamNames.Contains(name))
                        continue;

                    NDArrayList arrays = new NDArrayList();
                    foreach (var e in Execs)
                    {
                        arrays.Add(e.GradientArrays[i]);
                    }

                    GradArrays.Add(arrays);
                }
            }

            AuxArrays = new List<NDArrayList>();
            for (int i = 0; i < AuxNames.Length; i++)
            {
                var name = AuxNames[i];
                NDArrayList arrays = new NDArrayList();
                foreach (var e in Execs)
                {
                    arrays.Add(e.AuxiliaryArrays[i]);
                }

                AuxArrays.Add(arrays);
            }
        }

        public void BindExec(DataDesc[] data_shapes, DataDesc[] label_shapes, DataParallelExecutorGroup shared_group = null,
            bool reshape = false)
        {
            if(!reshape && Execs == null )
            {
                throw new Exception("reshape = false or Execs is null");
            }

            BatchSize = null;
            DataLayouts = DecideSlices(data_shapes);
            if (label_shapes != null)
                LabelLayouts = DecideSlices(label_shapes);

            for(int i = 0;i<Contexts.Length;i++)
            {
                var data_shapes_i = SlicedShape(data_shapes, i, DataLayouts);
                DataDesc[] label_shapes_i = new DataDesc[0];
                if (label_shapes != null)
                    label_shapes_i = SlicedShape(label_shapes, i, LabelLayouts);

                if(reshape)
                {
                    var newshapes = data_shapes_i.ToList();
                    newshapes.AddRange(label_shapes_i);
                    Execs[i] = _default_execs[i].Reshape(allow_up_sizing: true, newShapes: newshapes.ToArray());
                }
                else
                {
                    Execs.Add(BindiThExec(i, data_shapes_i, label_shapes_i, shared_group));
                }
            }

            DataShapes = data_shapes;
            LabelShapes = label_shapes;
            DataNames = data_shapes.Select(x => x.Name).ToArray();
            if (label_shapes != null)
                LabelNames = label_shapes.Select(x => x.Name).ToArray();

            CollectArrays();
        }

        public void Reshape(DataDesc[] data_shapes, DataDesc[] label_shapes)
        {
            if (DataShapes.Length == data_shapes.Length && LabelShapes.Length == label_shapes.Length)
                return;

            if (_default_execs == null)
                _default_execs = Execs.ToArray();

            BindExec(data_shapes, label_shapes, reshape: true);
        }

        public void SetParams(NDArrayDict arg_params, NDArrayDict aux_params, bool allow_extra = false)
        {
            foreach (var exec in Execs)
            {
                exec.CopyFromParams(arg_params, aux_params, allow_extra);
            }
        }

        public void GetParams(NDArrayDict arg_params, NDArrayDict aux_params)
        {
            Enumerable.Zip(ParamNames, ParamArrays, (name, block) =>
            {
                NDArray weight = null;
                foreach (var w in block)
                {
                    if (weight == null)
                    {
                        weight = w.ChangeContext(Context.Cpu());
                    }
                    else
                    {
                        weight += w.ChangeContext(Context.Cpu());
                    }
                }

                weight.AsType(arg_params[name].DataType).CopyTo(arg_params[name]);
                return true;
            });

            Enumerable.Zip(AuxNames, AuxArrays, (name, block) =>
            {
                NDArray weight = null;
                foreach (var w in block)
                {
                    if (weight == null)
                    {
                        weight = w.ChangeContext(Context.Cpu());
                    }
                    else
                    {
                        weight += w.ChangeContext(Context.Cpu());
                    }
                }

                weight.AsType(aux_params[name].DataType).CopyTo(aux_params[name]);
                return true;
            });
        }

        public void Forward(DataBatch data_batch, bool? is_train = null)
        {
            ExecutorGroup.LoadData(data_batch, DataArrays, DataLayouts);
            if (!is_train.HasValue)
                is_train = ForTraining;

            if (LabelArrays != null && data_batch.Label != null)
                ExecutorGroup.LoadLabel(data_batch, LabelArrays, LabelLayouts);

            foreach (var exec in Execs)
            {
                exec.Forward(isTrain: is_train.Value);
            }
        }

        public DataDesc[] GetOutputShapes()
        {
            var outputs = Execs[0].Outputs;
            var shapes = outputs.Select(x => x.Shape).ToArray();
            var out_names = Symbol.ListOutputs().ToArray();
            List<DataDesc> concat_shapes = new List<DataDesc>();
            for (int i = 0; i < out_names.Length; i++)
            {
                var key = out_names[i];
                var the_shape = shapes[i];
                var axis = OutputLayouts[i];
                if(axis >= 0)
                {
                    the_shape.Data[i] = BatchSize.Value;
                }

                concat_shapes.Add(new DataDesc(key, the_shape));
            }

            return concat_shapes.ToArray();
        }

        public List<NDArrayList> GetOutputs(bool merge_multi_context = true, int begin = 0, int? end = null)
        {
            if (end == null)
                end = NumOutputs;

            List<NDArrayList> outputs = new List<NDArrayList>();
            for (int i = begin; i < end; i++)
            {
                NDArrayList arrays = new NDArrayList();
                foreach (var exec in Execs)
                {
                    arrays.Add(exec.Outputs[i]);
                }

                outputs.Add(arrays);
            }

            if (merge_multi_context)
                return new List<NDArrayList>() { ExecutorGroup.MergeMultiContext(outputs, OutputLayouts) };

            return outputs;
        }

        public List<NDArrayList> GetStates(bool merge_multi_context = true)
        {
            if (merge_multi_context)
                throw new Exception("merge_multi_context=True is not supported for get_states yet");

            return StateArrays;
        }

        public void SetStates(List<NDArrayList> states = null, float? value = null)
        {
            if(states != null)
            {
                if (value.HasValue)
                    throw new Exception("Only one of states & value can be specified.");

                ExecutorGroup.LoadGeneral(states, StateArrays, states.Select(x => 0).ToArray());
            }
            else
            {
                if (!value.HasValue)
                    throw new Exception("At least one of states & value must be specified.");

                if (states != null)
                    throw new Exception("Only one of states & value can be specified.");

                foreach (var d_dst in StateArrays)
                {
                    for (int i = 0; i < d_dst.Length; i++)
                        d_dst[i] = nd.Full(value.Value, d_dst[i].Shape, d_dst[i].Context, d_dst[i].DataType);
                }
            }
        }

        public List<NDArrayList> GetInputGrads(bool merge_multi_context = true)
        {
            if (!InputsNeedGrad)
                throw new Exception("InputsNeedGrad is false");

            if (merge_multi_context)
                return new List<NDArrayList>() { ExecutorGroup.MergeMultiContext(InputGradArrays, DataLayouts) };

            return InputGradArrays;
        }

        public void Backward(NDArrayList out_grads = null)
        {
            if (!ForTraining)
                throw new Exception("Re-bind with for_training=True to run backward");

            if (out_grads == null)
                out_grads = new NDArrayList();

            for (int i = 0; i < Execs.Count; i++)
            {
                var exec_ = Execs[i];
                var islice = Slices[i];

                NDArrayList out_grads_slice = Enumerable.Zip(out_grads, OutputLayouts, (grad, axis) =>
                {
                    if(axis >= 0)
                    {
                        var og_my_slice = nd.SliceAxis(grad, axis, begin: islice.Begin, end: islice.End);
                        return og_my_slice;
                    }
                    else
                    {
                        return grad.ChangeContext(Contexts[i]);
                    }
                }).ToArray();

                exec_.Backward(out_grads_slice);
            }
        }

        public void UpdateMetric(EvalMetric eval_metric, NDArrayList labels, bool pre_sliced = false)
        {
            for (int current_exec = 0; current_exec < Execs.Count; current_exec++)
            {
                var texec = Execs[current_exec];
                var islice = Slices[current_exec];

                NDArrayList labels_slice = null;
                if (!pre_sliced)
                {
                    labels_slice = Enumerable.Zip(labels, OutputLayouts, (label, axis) =>
                    {
                        if (axis > 0)
                        {
                            var label_my_slice = nd.SliceAxis(label, axis, begin: islice.Begin, end: islice.End).AsInContext(label.Context);
                            return label_my_slice;
                        }
                        else
                        {
                            return label.Slice(islice.Begin, islice.End);
                        }
                    }).ToArray();
                }
                else
                {
                    labels_slice = labels[current_exec];
                }

                NDArrayDict labelDict = new NDArrayDict();
                NDArrayDict predDict = new NDArrayDict();
                int i = 0;
                foreach (var name in LabelNames.OrderBy(x => x).ToArray())
                {
                    labelDict.Add(name, labels_slice[i]);
                    i++;
                }

                i = 0;
                var outputs = texec.Outputs;
                foreach (var name in OutputNames.OrderBy(x => x).ToArray())
                {
                    predDict.Add(name, outputs[i]);
                    i++;
                }

                eval_metric.UpdateDict(labelDict, predDict);
            }
        }

        private Executor BindiThExec(int i, DataDesc[] data_shapes, DataDesc[] label_shapes,
            DataParallelExecutorGroup shared_group)
        {
            var shared_exec = shared_group == null ? null : shared_group.Execs[i];
            var context = Contexts[i];
            var shared_data_arrays = SharedDataArrays[i];
            var input_shapes = data_shapes.ToList();
            if (label_shapes != null)
                input_shapes.AddRange(label_shapes);

            var input_types = new Dictionary<string, DType>();
            foreach (var item in input_shapes)
            {
                input_types.Add(item.Name, item.DataType);
            }

            var group2ctx = Group2Ctxs[i];

            var executor = Symbol.SimpleBind(ctx: context, grad_req: GradReq,
                                           type_dict: input_types, stype_dict: null, group2ctx: group2ctx, shared_arg_names: ParamNames,
                                           shared_exec: shared_exec,
                                           shared_buffer: shared_data_arrays, input_shapes.ToArray());

            return executor;
        }

        private DataDesc[] SlicedShape(DataDesc[] shapes, int i, int[] major_axis)
        {
            var sliced_shape = Enumerable.Zip(shapes, major_axis, (desc, axis) =>
            {
                var shape = desc.Shape;
                if (axis >= 0)
                    shape.Data[axis] = Slices[i].End.Value - Slices[i].Begin; 

                return new DataDesc(desc.Name, shape, desc.DataType, desc.Layout);
            }).ToArray();

            return sliced_shape;
        }

        public void InstallMonitor(Monitor mon)
        {
            foreach (var exec in Execs)
            {
                mon.Install(exec);
            }
        }
    }
}