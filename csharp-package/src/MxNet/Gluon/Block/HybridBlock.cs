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
using MxNet.Interop;
using MxNet.Numpy;
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MxNet.Gluon
{
    public class CachedOpArg
    {
        public string Name;
        public bool IsArg;
        public int Index;
        public Parameter Param;

        public CachedOpArg((bool, string, int) tuple)
        {
            IsArg = tuple.Item1;
            Index = tuple.Item3;
            Name = tuple.Item2;
        }

        public CachedOpArg((bool, string, Parameter) tuple)
        {
            IsArg = tuple.Item1;
            Param = tuple.Item3;
            Name = tuple.Item2;
        }

        public void ResetCtx(Context ctx)
        {
            if (Param != null)
                Param.ResetCtx(ctx);
        }
    }

    public class HybridBlock : Block
    {
        internal CachedOp _cached_op;
        internal readonly List<CachedOpArg> _cached_op_args = new List<CachedOpArg>();
        internal Dictionary<string, object> _flags = new Dictionary<string, object>();
        internal bool _called_infer_shape_already;
        internal string _backend;
        internal Dictionary<string, string> _backend_opts;
        internal bool _v2;
        internal bool _partition_if_dynamic;
        internal bool _first_forward;

        public HybridBlock() : base()
        {
            this._v2 = true;
            this._cached_graph = (null, null);
            this._cached_op = null;
            this._out_format = null;
            this._in_format = null;
            this._called_infer_shape_already = false;
            this._active = false;
            this._flags = new Dictionary<string, object>();
            this._callback = null;
            this._monitor_all = false;
            this._backend = null;
            this._backend_opts = new Dictionary<string, string>();
            this._partition_if_dynamic = true;
            this._first_forward = true;
        }

        public HybridBlock(Dictionary<string, Block> blocks, bool loadkeys = false)
            : this()
        {
            foreach (var item in blocks)
            {
                if(loadkeys)
                    RegisterChild(item.Value, item.Key);
                else
                    RegisterChild(item.Value);
            }
        }

        private (SymbolList, _Symbol) GetGraphV1(NDArrayOrSymbolList args)
        {
            if (_cached_graph == null)
            {
                var inputs = new SymbolList();
                var (flatten_args, _in_format) = Flatten(args, "input");

                var flatten_inputs = new List<NDArrayOrSymbolList>();
                var symbol_inputs = new List<_Symbol>();
                var cnt = 0;
                var real_arg_num = flatten_args.Select(x => x != null).Count();
                if (real_arg_num == 0)
                {
                    throw new Exception($"All args are None and we do not support such a case.");
                }

                foreach (var arg in flatten_args)
                {
                    _Symbol arg_sym;
                    if (arg != null)
                    {
                        if (real_arg_num > 1)
                        {
                            arg_sym = _Symbol.Var("datacnt{}");
                        }
                        else
                        {
                            arg_sym = _Symbol.Var("data");
                        }

                        cnt += 1;
                        flatten_inputs.Add(new NDArrayOrSymbolList { arg_sym });
                        symbol_inputs.Add(arg_sym);
                    }
                    else
                    {
                        flatten_inputs.Add(null);
                    }
                }

                var grouped_inputs = Regroup(flatten_inputs, this._in_format).Item1;
                var outputs = new NDArrayOrSymbolList();
                using (var _ = new _BlockScope(this))
                {
                    var @params = new SymbolDict();
                    foreach (var item in _reg_params) @params[item.Key] = item.Value.Var();

                    foreach (var input in grouped_inputs)
                        outputs.Add(HybridForward((input, new NDArrayOrSymbol(@params.Values))));
                }

                var (@out, _out_format) = Flatten(outputs, "output");
                _cached_graph = (inputs, _Symbol.Group(@out.ToSymbols()));
            }

            return _cached_graph.Value;
        }

        private (SymbolList, _Symbol) GetGraphV2(NDArrayOrSymbolList args)
        {
            List<string> arg_names = new List<string>();
            if (_cached_graph == null)
            {
                var inputs = new SymbolList();
                var (flatten_args, _in_format) = Flatten(args, "input");
                flatten_args = new NDArrayOrSymbolList((from ele in flatten_args
                                                        select ele != null ? ele.NdX.Detach() : null).ToArray());
                NDArrayOrSymbolList real_args = (from ele in flatten_args
                                 where ele != null
                                 select ele).ToList();
                if (real_args.Length == 0)
                {
                    throw new Exception("All args are None and we do not support such a case.");
                }
                if (real_args.Length == 1)
                {
                    arg_names = new List<string> { "data" };
                }
                else
                {
                    for (int i = 0; i < real_args.Length; i++)
                    {
                        arg_names.Add($"data{i}");
                    }
                }

                SymbolList symbol_inputs = new SymbolList();
                for(int i = 0; i< real_args.Length; i++)
                {
                    var name = arg_names[i];
                    var arg = real_args[i];
                    symbol_inputs.Add(_Symbol.Var(name));
                }

                DeferredCompute.SetVariable(real_args.NDArrays, symbol_inputs);
                args = Regroup(new List<NDArrayOrSymbolList> { flatten_args }, this._in_format).Item1;
                NDArrayOrSymbolList @out;
                using(var ag = Autograd.Pause())
                {
                    DeferredCompute.Context();
                    @out = base.Call(args);
                }

                var (flatten_out, out_format) = Flatten(@out, "output");
                this._out_format = out_format.ToList();
                var symbol_outputs = DeferredCompute.GetSymbol(flatten_out.NDArrays);
                this._cached_graph = (symbol_inputs, symbol_outputs);
            }
            return this._cached_graph.Value;
        }

        private (SymbolList, _Symbol) GetGraph(NDArrayOrSymbolList args)
        {
            if (_cached_graph == null)
            {
                if (!this._v2)
                {
                    return this.GetGraphV1(args);
                }
                else
                {
                    // Gluon 2 based on deferred compute mode
                    return this.GetGraphV2(args);
                }
            }

            return this._cached_graph.Value;
        }

        private void BuildCache(NDArrayOrSymbolList args, bool update_graph = true)
        {
            var (data, @out) = GetGraph(args);
            var data_names = new Dictionary<string, int>();
            for(int i = 0; i < args.Length; i++)
            {
                data_names.Add(data[i].Name, i);
            }
            
            var @params = this.CollectParams().Values().ToDictionary(p => p.Var().Name, p => p);
            var param_serialization_names = this.CollectParams().Items().ToDictionary(_tup_3 => _tup_3.Value.Var().Name, _tup_3 => _tup_3.Key);
            var param_names = new HashSet<string>(@params.Keys).ToArray();

            param_names = MxUtil.Set(@params.Keys.ToList()).ToArray();
            var input_names = @out.ListInputs().ToArray();
            var expected_names = MxUtil.Set(input_names.ToList());
            foreach (var name in expected_names)
                if (!param_names.Contains(name) && !data_names.ContainsKey(name))
                    throw new Exception($"Unknown input to HybridBlock: {name}");

            var used_data_names = new List<string>();
            var unused_data_names = new List<string>();

            var used_param_names = new List<string>();
            var unused_param_names = new List<string>();

            foreach (var name in data_names.Keys)
            {
                if (expected_names.Contains(name))
                    used_data_names.Add(name);

                if (!expected_names.Contains(name))
                    unused_data_names.Add(name);
            }

            if (used_data_names.Count != data_names.Count)
                Logger.Warning($"The {string.Join(",", unused_data_names)} input to HybridBlock is not used by any " +
                               "computation. Is this intended?");

            foreach (var name in param_names)
            {
                if (expected_names.Contains(name))
                    used_param_names.Add(name);

                if (!expected_names.Contains(name))
                    unused_param_names.Add(name);
            }

            if (used_param_names.Count != param_names.Length)
                Logger.Warning($"The {string.Join(",", used_param_names)} input to HybridBlock is not used by any " +
                               "computation. Is this intended?");

            var _tup_5 = Flatten(args, "input");
            args = _tup_5.Item1;
            try
            {
                foreach (var name in input_names)
                {
                    if (@params.ContainsKey(name)) 
                    {
                        @params[name].Data();
                    }
                }
            }
            catch (Exception ex)
            {
                this.DeferredInferShape(args);
                foreach (var name in input_names)
                {
                    if (@params.ContainsKey(name)) {
                        @params[name].FinishDeferredInit();
                    }
                }
            }

            var arg_dict = new NDArrayDict();
            var aux_dict = new NDArrayDict();
            if (!string.IsNullOrWhiteSpace(this._backend))
            {
                // set context for inputs
                var _tup_6 = GatherTypeCtxInfo(args);
                var ctx_set = _tup_6.Item3;
                Context ctx = null;
                if(ctx_set.Length > 0)
                {
                    ctx = ctx_set.Last();
                    ctx_set.ToList().RemoveAt(ctx_set.Length - 1);
                }
                
                // get list of params in the order of out.list_arguments
                var input_shapes = new Dictionary<string, Shape>();
                foreach (var name in @out.ListArguments())
                {
                    if (data_names.ContainsKey(name) && data_names[name] < args.Length)
                    {
                        if (args[data_names[name]].IsNDArray)
                        {
                            arg_dict[name] = args[data_names[name]];
                        }
                        else if (args[data_names[name]].IsSymbol)
                        {
                            var shape_str = args[data_names[name]].SymX.ListAttr()["__shape__"];
                            int[] shape_ints = shape_str.Replace("(", "").Replace(")", "").Split(',').Select(x => Convert.ToInt32(x.Trim())).ToArray();
                            input_shapes[name] = new Shape(shape_ints);
                        }
                    }
                    else if (@params.ContainsKey(name)) {
                        arg_dict[name] = @params[name].Data();
                    }
                }

                foreach (var name in @out.ListAuxiliaryStates())
                {
                    if (data_names.ContainsKey(name) && data_names[name] < args.Length)
                    {
                        if (args[data_names[name]].IsNDArray)
                        {
                            aux_dict[name] = args[data_names[name]];
                        }
                        else if (args[data_names[name]].IsSymbol && args[data_names[name]].SymX.ListAttr().ContainsKey("__shape__"))
                        {
                            var shape_str = args[data_names[name]].SymX.ListAttr()["__shape__"];
                            int[] shape_ints = shape_str.Replace("(", "").Replace(")", "").Split(',').Select(x => Convert.ToInt32(x.Trim())).ToArray();
                            input_shapes[name] = new Shape(shape_ints);
                        }
                    }
                    else if (@params.ContainsKey(name)) {
                        aux_dict[name] = @params[name].Data();
                    }
                }

                // Partition the graph
                // ToDo: Missing backendOpts parameters
                @out = @out.OptimizeFor(this._backend, arg_dict, aux_dict, ctx, input_shapes);
                //update cached graph with partitioned graph
                if (update_graph)
                {
                    this._cached_graph = (data, @out);
                }
            }

            input_names = @out.ListInputs().ToArray();

            var data_indices = new List<int>();
            var param_indices = new List<int>();

            for (var i = 0; i < input_names.Length; i++)
            {
                Parameter param = null;
                ndarray param_data;
                var name = input_names[i];
                (bool, string, Parameter) triple = (false, "", param);
                if (data_names.ContainsKey(name))
                {
                    data_indices.Add(i);
                    triple = (true, name, @params[name]);
                }
                else
                {
                    param_indices.Add(i);
                    
                    string serialization_name;
                    if (@params.ContainsKey(name)) {
                        param = @params[name];
                        serialization_name = param_serialization_names[name];
                    } 
                    else
                    {
                        // The param is missing from the original params dictionary, which means the param must have
                        // been added by the Partition API backend
                        if (arg_dict.Contains(name) || !string.IsNullOrWhiteSpace(name))
                        {
                            param_data = arg_dict[name];
                        }
                        else if (aux_dict.Contains(name))
                        {
                            param_data = aux_dict[name];
                        }
                        else
                        {
                            throw new Exception("A parameter was added to the graph during optimization but it was not added to the parameter dicts.\nPlease check the backend.");
                        }

                        param = new Parameter(name, dtype: param_data.dtype);
                        param._var_name = name;
                        serialization_name = name;
                        param.LoadInit(param_data, new Context[] { param_data.ctx });
                    }

                    triple = (false, serialization_name, param);
                }

                _cached_op_args.Add(new CachedOpArg(triple));
            }

            var flags = new Dictionary<string, string>()
            {
                { "data_indices", "[" + string.Join(",", data_indices.Select(i => i.ToString()).ToArray()) + "]" },
                { "param_indices", "[" + string.Join(",", param_indices.Select(i => i.ToString()).ToArray()) + "]" }
            };

            foreach (var item in _flags) flags.Add(item.Key, item.Value.ToString());

            _cached_op = new CachedOp(@out, flags);
        }

        private void DeferredInferShape(NDArrayOrSymbolList args)
        {
            try
            {
                InferShape(args);
            }
            catch(Exception ex)
            {
                throw new Exception("Deferred initialization failed because shape cannot be inferred: " + ex.Message);
            }
        }

        internal NDArrayOrSymbolList CallCachedOp(NDArrayOrSymbolList args)
        {
            if (_cached_op == null)
                BuildCache(args);

            if (this._first_forward && this._partition_if_dynamic)
            {
                this._first_forward = false;
                // partition static shape ops if the graph contains any dynamic shape op
                var _tup_1 = this._cached_graph.Value;
                var is_dynamic = _tup_1.Item2.HasDynamicShapeOp();
                if (is_dynamic)
                {
                    this._backend = "static_shape";
                    this._backend_opts = this._flags.ToDictionary(_tup_2 => _tup_2.Key, _tup_2 => _tup_2.Value.ToString());
                    this.BuildCache(args, update_graph: false);
                }
            }

            Debug.Assert(this._cached_op != null, "Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/deepakkumar1984/MxNet.Sharp.");
            if (this._callback != null)
            {
                this._cached_op.RegisteropHook(this._callback, this._monitor_all);
                if (this._flags.Count >= 2 && (this._flags[this._flags.Keys.ToArray()[1]] != null || this._flags[this._flags.Keys.ToArray()[0]] != null))
                {
                    Logger.Warning("register_op_hook is experimental when static_alloc=True / static_shape=True  and may not work correctly");
                }
            }
            var _tup_3 = Flatten(args, "input");
            args = _tup_3.Item1;
            bool valid = false;
            var fmt = _tup_3.Item2;
            if (fmt.Length != this._in_format.Count)
            {
                // Do not raise in the case that the fmt or stored_fmt ends with None and
                // We are relying on the default values.
                if (this._in_format.Count > fmt.Length)
                {
                    valid = (from i in Enumerable.Range(fmt.Length, this._in_format.Count - fmt.Length)
                             select (this._in_format[i] == -1)).All(x => x);
                    valid = valid && fmt == this._in_format.Take(fmt.Length);
                }
                else if (this._in_format.Count < fmt.Length)
                {
                    valid = ((from i in Enumerable.Range(this._in_format.Count, fmt.Length - this._in_format.Count)
                              select (fmt[i] == -1)).ToList()).All(x => x);
                    valid = valid && fmt.Take(this._in_format.Count) == this._in_format;
                }
                else
                {
                    valid = false;
                }

                if (!valid)
                {
                    throw new Exception($"The argument structure of HybridBlock does not match the cached version. Stored format = {string.Join(",", fmt)}, input format = {string.Join(",", _in_format)}");
                }
            }

            var args_without_none = (from ele in args
                                     where ele != null
                                     select ele).ToList();
            var cargs = (from _tup_4 in this._cached_op_args
                         let is_arg = _tup_4.IsArg
                         let name = _tup_4.Name
                         select is_arg ? args_without_none[_tup_4.Index].NdX : _tup_4.Param.Data()).ToList();

            var @out = _cached_op.Call(cargs);
            return Regroup(new List<NDArrayOrSymbolList> { @out.NDArrayOrSymbols }, _out_format).Item1;
        }

        public void OptimizeFor(ndarray x, string backend = null, bool clear = false, bool partition_if_dynamic = true, bool static_alloc = false,
               bool static_shape = false, int inline_limit = 2, int? forward_bulk_size = null, int? backward_bulk_size = null, Dictionary<string, string> backend_opts = null, NDArrayList args = null)
        {
            this._backend = backend;
            if (backend_opts != null && backend_opts.Count > 0)
            {
                this._backend_opts = backend_opts;
            }
            if (clear || !this._active)
            {
                this.Hybridize(true, partition_if_dynamic, static_alloc, static_shape, inline_limit, forward_bulk_size, backward_bulk_size);
            }
            NDArrayOrSymbolList inputs = new NDArrayOrSymbolList()
            {
                x
            };

            if (args != null)
                inputs.Add(args);

            // do part of forward API call
            var _tup_1 = GatherTypeCtxInfo(inputs);
            var has_symbol = _tup_1.Item1;
            var has_ndarray = _tup_1.Item2;
            var ctx_set = _tup_1.Item3;
            if (!has_symbol && !has_ndarray)
            {
                throw new Exception("In HybridBlock, there must be one ndarray or one _Symbol in the input. Please check the type of the args.\n");
            }
            if (ctx_set.Length > 1)
            {
                throw new Exception($"Found multiple contexts in the input, After hybridized, the HybridBlock only supports one input context. " +
                    $"You can print the ele.ctx in the input arguments to inspect their contexts. Find all contexts = {string.Join(", ", ctx_set.Select(i => i.ToString()))}");
            }

            this.BuildCache(inputs);
            Debug.Assert(this._cached_op != null, "Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/deepakkumar1984/MxNet.Sharp.");
            // do not actually call the cached_op
            this._first_forward = true;
            // clear the backend
            this._backend = null;
            this._backend_opts = new Dictionary<string, string>();
        }

        public virtual void ClearCachedOp()
        {
            _cached_graph = null;
            _cached_op = null;
            this._first_forward = true;
        }

        public override void RegisterChild(Block block, string name = null)
        {
            if (block is HybridBlock) {
                base.RegisterChild(block, name);
                if (this._active)
                {
                    Logger.Warning("Currently the model has been hybridized. Automatically deactivate the hybridization when adding new children block.");
                    this._active = false;
                }

                this.ClearCachedOp();
            }
            else
                throw new Exception("Children of HybridBlock must also be HybridBlock, " +
                                    $"but {block.Alias()} has type {block.GetType().Name}. If you are using Sequential, " +
                                    "please try HybridSequential instead.");
        }

        public override void Hybridize(bool active = true, bool partition_if_dynamic = true, bool static_alloc = false, bool static_shape = false,
            int inline_limit = 2, int? forward_bulk_size = null, int? backward_bulk_size = null)
        {
            _active = active;
            this._partition_if_dynamic = partition_if_dynamic;
            this._flags = new Dictionary<string, object> {
                { "static_alloc", static_alloc },
                { "static_shape", static_shape },
                { "inline_limit", inline_limit }
            };

            if (forward_bulk_size != null)
            {
                this._flags.Add("forward_bulk_size", forward_bulk_size);
            }
            if (backward_bulk_size != null)
            {
                this._flags.Add("backward_bulk_size", backward_bulk_size);
            }

            ClearCachedOp();
            if (active && _forward_hooks != null || _forward_pre_hooks != null)
                Logger.Warning($"{this.GetType().Name} is being hybridized while still having forward hook/pre-hook. If \"{this.GetType().Name}\" is a child of HybridBlock, the hooks will not take effect.");

            base.Hybridize(active, static_alloc: static_alloc, static_shape: static_shape, inline_limit: inline_limit, forward_bulk_size: forward_bulk_size, backward_bulk_size: backward_bulk_size);
        }

        public override void Cast(DType dtype)
        {
            if (this._active)
            {
                Logger.Warning("Currently the model has been hybridized. Automatically deactivate the hybridization when cast the block to use another data type.");

                this._active = false;
            }

            ClearCachedOp();
            base.Cast(dtype);
        }

        private void InterAttrs(string infer_fn, string attr, NDArrayOrSymbolList arguments)
        {
            var (inputs, @out) = GetGraph(arguments);
            var (args, _) = Flatten(arguments, "input");
            var args_without_none = (from ele in args
                                     where ele != null
                                     select ele).ToArray();
            if (infer_fn == "infer_shape")
            {
                var args_shape = new Dictionary<string, Shape>();
                var sdict = new Dictionary<string, Shape>();
                for (var i = 0; i < args_without_none.Length; i++) args_shape.Add(inputs[i].Name, args_without_none[i].NdX.shape);

                var (arg_attrs, _, aux_attrs) = @out.InferShape(args_shape);
                if (arg_attrs == null)
                    throw new Exception("No Args shape found");

                var arg_names = @out.ListArguments().ToArray();
                for (var i = 0; i < arg_attrs.Length; i++) sdict.Add(arg_names[i], new Shape(arg_attrs[i].Data));

                var aux_names = @out.ListAuxiliaryStates().ToArray();
                for (var i = 0; i < aux_attrs.Length; i++) sdict[aux_names[i]] = aux_attrs[i];

                var collectedValues = CollectParams().Values();
                for (var i = 0; i < collectedValues.Length; i++)
                    collectedValues[i]._shape = sdict[collectedValues[i]._var_name];
            }
            else if (infer_fn == "infer_type")
            {
                var args_shape = new Dictionary<string, DType>();
                var sdict = new Dictionary<string, DType>();
                for (var i = 0; i < args_without_none.Length; i++) args_shape.Add(inputs[i].Name, args_without_none[i].NdX.dtype);

                var (arg_attrs, _, aux_attrs) = @out.InferType(args_shape);
                if (arg_attrs == null)
                    throw new Exception("No Args shape found");

                var arg_names = @out.ListArguments().ToArray();
                for (var i = 0; i < arg_attrs.Length; i++) sdict.Add(arg_names[i], arg_attrs[i]);

                var aux_names = @out.ListAuxiliaryStates().ToArray();
                for (var i = 0; i < aux_attrs.Length; i++) sdict[aux_names[i]] = aux_attrs[i];

                var collectedValues = CollectParams().Values();
                for (var i = 0; i < collectedValues.Length; i++)
                    collectedValues[i].DataType = sdict[collectedValues[i]._var_name];
            }
        }

        public void InferShape(NDArrayOrSymbolList args)
        {
            if (!this._v2)
            {
                InterAttrs("infer_shape", "shape", args);
            }
            else
            {
                var @params = (from p in this._reg_params.Values()
                              where !Utils.ShapeIsKnown(p.Shape)
                              select p).ToList();

                if (@params.Count > 0) {
                    var params_str = string.Join(", ", from p in @params select $"{p.Name} ({p.Shape})");
                    throw new Exception($"{this.Alias()} has parameters with unknown shape. You need to either specify the shape in __init__ or implement {this.Alias()}.infer_shape to set the parameter shapes based on the first input. Parameters with unknown shapes are {params_str}");
                }
            }
        }

        public void InferType(NDArrayOrSymbolList args)
        {
            InterAttrs("infer_type", "dtype", args);
        }

        public (string, string) Export(string path, int epoch = 0, bool remove_amp_cast = true)
        {
            if (_cached_graph == null)
                throw new Exception("Please first call block.hybridize() and then run forward with " +
                                    "this block at least once before calling export.");

            var sym = _cached_graph.Value.Item2.ShallowCopy();
            var @params = this.CollectParams();
            // In export we have global information on the structure of the graph
            // can rename the symbol inputs to human-readable, deterministic names.
            // That's not true in general, which is why internally random unique identifiers are used.
            var rename_map = @params.ToDictionary(_tup_3 => _tup_3.Value.Var().Name, _tup_3 => _tup_3.Key);
            foreach (var var in sym.GetInputs())
            {
                if (rename_map.ContainsKey(var.Name))
                {
                    var.SetAttr(new Dictionary<string, string>() { { "name", rename_map[var.Name] } });
                }
            }

            var sym_filename = String.Format("%s-symbol.json", path != null ? path : "");
            if(path != null)
                sym.Save($"{path}\\symbol.json", remove_amp_cast);

            var arg_names = new HashSet<string>(sym.ListArguments()).ToList();
            var aux_names = new HashSet<string>(sym.ListAuxiliaryStates()).ToList();
            var arg_dict = new NDArrayDict();
            foreach (var _tup_4 in this._cached_op_args)
            {
                var is_arg = _tup_4.IsArg;
                var name = _tup_4.Name;
                var param = _tup_4.Param;
                if (!is_arg)
                {
                    if (arg_names.Contains(name))
                    {
                        arg_dict[$"arg:{name}"] = param.Reduce();
                    }
                    else if (!aux_names.Contains(name))
                    {
                        Logger.Warning($"Parameter \"{name}\" is not found in the graph. ");
                    }
                    else
                    {
                        arg_dict[$"aux:{name}"] = param.Reduce();
                    }
                }
            }

            var params_filename = String.Format("%s-%04d.params", path != null ? path : "", epoch);
            if (path != null)
            {
                ndarray.Save(params_filename, arg_dict);
                return (sym_filename, arg_dict.Count > 0 ? params_filename : null);
            }

            return ("", "");
        }

        public (_Symbol, NDArrayDict) Export(int epoch = 0, bool remove_amp_cast = true)
        {
            if (_cached_graph == null)
                throw new Exception("Please first call block.hybridize() and then run forward with " +
                                    "this block at least once before calling export.");

            var sym = _cached_graph.Value.Item2.ShallowCopy();
            var @params = this.CollectParams();
            // In export we have global information on the structure of the graph
            // can rename the symbol inputs to human-readable, deterministic names.
            // That's not true in general, which is why internally random unique identifiers are used.
            var rename_map = @params.ToDictionary(_tup_3 => _tup_3.Value.Var().Name, _tup_3 => _tup_3.Key);
            foreach (var var in sym.GetInputs())
            {
                if (rename_map.ContainsKey(var.Name))
                {
                    var.SetAttr(new Dictionary<string, string>() { { "name", rename_map[var.Name] } });
                }
            }

            var arg_names = new HashSet<string>(sym.ListArguments()).ToList();
            var aux_names = new HashSet<string>(sym.ListAuxiliaryStates()).ToList();
            var arg_dict = new NDArrayDict();
            foreach (var _tup_4 in this._cached_op_args)
            {
                var is_arg = _tup_4.IsArg;
                var name = _tup_4.Name;
                var param = _tup_4.Param;
                if (!is_arg)
                {
                    if (arg_names.Contains(name))
                    {
                        arg_dict[$"arg:{name}"] = param.Reduce();
                    }
                    else if (!aux_names.Contains(name))
                    {
                        Logger.Warning($"Parameter \"{name}\" is not found in the graph. ");
                    }
                    else
                    {
                        arg_dict[$"aux:{name}"] = param.Reduce();
                    }
                }
            }

            if (remove_amp_cast)
            {
                NativeMethods.MXSymbolRemoveAmpCast(sym.GetHandle(), out var handle);
                sym = new _Symbol(handle);
            }

            return (sym, arg_dict);
        }

        public override void RegisterOpHook(Action<string, string, ndarray> callback, bool monitor_all = false)
        {
            Action<string, string, ndarray> c_callback = (name, op_name, array) => {
                callback(name, op_name, array);
            };

            this._callback = c_callback;
            this._monitor_all = monitor_all;
            foreach (var cld in this._childrens.Values)
            {
                cld._callback = c_callback;
                cld._monitor_all = monitor_all;
            }
        }

        public override NDArrayOrSymbolList Forward(NDArrayOrSymbolList inputs)
        {
            var @params = new Dictionary<string, NDArrayOrSymbol>();
            var (has_symbol, has_ndarray, ctx_set, first_ctx)  = GatherTypeCtxInfo(inputs);
            if (has_symbol && has_ndarray)
            {
                throw new Exception("In HybridBlock, we do not support mixed NDArrays and Symbols types for the input. Please check the type of the args.\n");
            }

            if (!has_symbol && !has_ndarray)
            {
                throw new Exception("In HybridBlock, there must be one ndarray or one _Symbol in the input. Please check the type of the args.\n");
            }

            if (has_ndarray)
            {
                var ctx = first_ctx;
                if (_active && !DeferredCompute.IsDeferredCompute())
                {
                    if (ctx_set.Length > 1)
                    {
                        throw new Exception($"Find multiple contexts in the input, After hybridized, the HybridBlock only supports one input context. " +
                            $"You can print the ele.ctx in the input arguments to inspect their contexts. Find all contexts = {string.Join(",", ctx_set.Select(i => i.ToString()))}");
                    }

                    return CallCachedOp(inputs);
                }

                try
                {
                    foreach (var p in _reg_params) @params[p.Key] = p.Value.Data(ctx);
                }
                catch (DeferredInitializationException ex)
                {
                    @params.Clear();
                    DeferredInferShape(inputs);
                    foreach (var p in Params.Items()) p.Value.FinishDeferredInit();

                    foreach (var p in _reg_params) @params[p.Key] = p.Value.Data(ctx);
                }

                inputs.Add(@params.Values.ToList());
                return HybridForward(inputs);
            }

            using (var b = new _BlockScope(this))
            {
                foreach (var p in _reg_params) @params[p.Key] = p.Value.Var();
                inputs.Add(@params.Values.ToList());
                return HybridForward(inputs);
            }
        }

        public override NDArrayOrSymbolList Call(NDArrayOrSymbolList inputs)
        {
            if (!this._v2)
            {
                // Gluon 1 based on F:  hybrid_forward is defined by user
                return base.Call(inputs);
            }
            else
            {
                // Gluon 2 based on deferred compute mode
                //Debug.Assert(this.forward != HybridBlock.forward);
                //Debug.Assert("Must either define {name}.forward or {name}.hybrid_forward. Defining {name}.hybrid_forward is deprecated.".format(name: type(this).@__name__));
             
                this.InferShape(inputs);

                if (!this._called_infer_shape_already)
                {
                    foreach (var p in this._reg_params.Values())
                    {
                        p.FinishDeferredInit();
                    }

                    this._called_infer_shape_already = true;
                }
                if (!this._active)
                {
                    // Normal imperative computation of forward()
                    return base.Call(inputs);
                }

                if (DeferredCompute.IsDeferredCompute())
                {
                    // Deferred compute is already enabled. This typically means that the current
                    // HybridBlock is a child block of a HybridBlock that has been hybridized.
                    return base.Call(inputs);
                }

                return this.CallCachedOp(inputs).FirstOrDefault();
            }
        }


        public virtual NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            return args;
        }

        public override void ResetCtx(Context ctx)
        {
            var @params = this.CollectParams();
            if (this._cached_op != null)
            {
                foreach (var p in this._cached_op_args)
                {
                    // resetting parameters creating by the partitioning backend
                    if (!@params.Contains(p.Name)) {
                        p.ResetCtx(ctx);
                    }
                }
            }
            foreach (var p in @params.Values()) {
                p.ResetCtx(ctx);
            }
        }
    }
}