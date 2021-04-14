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
using System.Text;
using MxNet.IO;
using NumpyDotNet;

namespace MxNet
{
    public class ExecuterManager
    {
        internal static Slice[] SplitInputSlice(int batch_size, int[] work_load_list)
        {
            var total_work_load = work_load_list.Sum();
            var batch_num_list = new int[work_load_list.Length];
            for (var i = 0; i < work_load_list.Length; i++)
                batch_num_list[i] = (int) Math.Round((double) work_load_list[i] * batch_size / total_work_load);

            var slices = new List<Slice>();
            var end = 0;
            var begin = 0;
            foreach (var batch_num in batch_num_list)
            {
                begin = Math.Min(end, batch_size);
                end = Math.Min(begin + batch_num, batch_size);
                if (begin >= end)
                    throw new Exception("Too many slices. Some splits are empty.");

                slices.Add(new Slice(begin, end));
            }

            return slices.ToArray();
        }

        internal static void CheckArguments(Symbol symbol)
        {
            var arg_set = new List<string>();
            var arg_names = symbol.ListArguments();
            foreach (var name in arg_names)
            {
                if (arg_set.Contains(name))
                    throw new Exception(string.Format("Find duplicated argument name \"{0}\", " +
                                                      "please make the weight name non-duplicated(using name arguments)," +
                                                      " arguments are {1}", name,
                        string.Join(",", arg_names.ToArray())));

                arg_set.Add(name);
            }

            var aux_set = new List<string>();
            var aux_names = symbol.ListAuxiliaryStates();
            foreach (var name in aux_names)
            {
                if (aux_set.Contains(name))
                    throw new Exception(string.Format("Find duplicated auxiliary param name \"{0}\", " +
                                                      "please make the weight name non-duplicated(using name arguments)," +
                                                      " arguments are {1}", name,
                        string.Join(",", arg_names.ToArray())));

                aux_set.Add(name);
            }
        }

        internal static void LoadGeneral(NDArrayList data, NDArrayList targets)
        {
            for (var i = 0; i < data.Length; i++)
            {
                var d_src = data[i];
                var d_targets = targets[i];
                d_src.CopyTo(d_targets);
            }
        }

        internal static void LoadData(DataBatch batch, NDArrayList targets)
        {
            LoadGeneral(batch.Data, targets);
        }

        internal static void LoadLabel(DataBatch batch, NDArrayList targets)
        {
            LoadGeneral(batch.Label, targets);
        }

        internal static Executor BindExec(Symbol sym, Context ctx, Dictionary<string, Shape> input_shapes,
            string[] param_names, bool need_grad = false,
            Executor base_exec = null, NDArrayDict shared_data_arrays = null,
            Dictionary<string, DType> input_types = null, Logger logger = null)
        {
            var (arg_shape, _, aux_shape) = sym.InferShape(input_shapes);
            if (arg_shape == null)
                throw new ArgumentNullException("arg_shape");

            if (input_types == null)
            {
                input_types = new Dictionary<string, DType>();
                foreach (var item in input_shapes.Keys) input_types.Add(item, DType.Float32);
            }

            var (arg_types, _, aux_types) = sym.InferType(input_types);

            if (arg_types == null)
                throw new ArgumentNullException("arg_types");

            var arg_arrays = new NDArrayList();
            var aux_arrays = new NDArrayList();
            var grad_arrays = need_grad ? new NDArrayDict() : null;

            var arg_names = sym.ListArguments();
            var needGradSet = new List<string>();
            if (!need_grad)
            {
                needGradSet = new List<string>();
            }
            else
            {
                foreach (var item in arg_names)
                    if (!input_shapes.ContainsKey(item))
                        needGradSet.Add(item);

                needGradSet = MxUtil.Set(needGradSet);
            }

            var grad_req = new Dictionary<string, OpGradReq>();
            foreach (var item in arg_names)
                if (needGradSet.Contains(item))
                    grad_req.Add(item, OpGradReq.Write);

            for (var i = 0; i < arg_names.Count; i++)
            {
                var name = arg_names[i];
                NDArray arg_arr = null;
                NDArray grad_arr = null;
                if (!param_names.Contains(name))
                {
                    if (shared_data_arrays != null && shared_data_arrays.Contains(name))
                    {
                        arg_arr = shared_data_arrays[name];
                        if (arg_arr.Shape.Size >= arg_shape[i].Size)
                        {
                            if (arg_types[i].Name != arg_arr.DataType.Name)
                                throw new ArgumentException("arg_type and arg_arr datatype mismatch");

                            arg_arr = arg_arr.Reshape(arg_shape[i]);
                        }
                        else
                        {
                            var logmsg = new StringBuilder();
                            logmsg.AppendFormat("bucketing: data \"{0}\" has a shape {1}", name, arg_shape[i]);
                            logmsg.AppendFormat(", which is larger than already allocated ");
                            logmsg.AppendFormat("shape {0}", arg_arr.Shape);
                            logmsg.AppendFormat(". Need to re-allocate. Consider putting default_bucket_key " +
                                                "to be the bucket taking the largest input for better memory sharing.");

                            Logger.Warning(logmsg.ToString());

                            arg_arr = nd.Zeros(arg_shape[i], ctx, arg_types[i]);
                            shared_data_arrays[name] = arg_arr;
                        }
                    }
                    else
                    {
                        arg_arr = nd.Zeros(arg_shape[i], ctx, arg_types[i]);
                        if (shared_data_arrays != null)
                            shared_data_arrays[name] = arg_arr;
                    }

                    arg_arrays.Add(arg_arr);
                }
                else
                {
                    if (base_exec == null)
                    {
                        arg_arr = nd.Zeros(arg_shape[i], ctx, arg_types[i]);
                        if (needGradSet.Contains(name))
                        {
                            grad_arr = nd.Zeros(arg_shape[i], ctx, arg_types[i]);
                            grad_arrays[name] = grad_arr;
                        }
                        else
                        {
                            arg_arr = base_exec.ArgmentDictionary()[name];
                            if (arg_arr.Shape != arg_shape[i])
                                throw new ArgumentException("arg_arr.Shape != arg_shape[i]");

                            if (arg_arr.DataType != arg_types[i])
                                throw new ArgumentException("arg_arr.DataType != arg_types[i]");

                            if (needGradSet.Contains(name))
                                grad_arrays[name] = base_exec.GradientDictionary()[name];
                        }

                        arg_arrays.Add(arg_arr);
                    }
                }
            }

            if (base_exec != null)
                for (var i = 0; i < aux_shape.Length; i++)
                {
                    var s = aux_shape[i];
                    var t = aux_types[i];
                    aux_arrays.Add(nd.Zeros(s, ctx, t));
                }
            else
                foreach (var item in base_exec.AuxiliaryDictionary())
                    aux_arrays.Add(item.Value);

            var executor = sym.Bind(ctx, arg_arrays, grad_arrays.Values.ToList(), grad_req.Values.ToList(), aux_arrays);
            return executor;
        }
    }
}