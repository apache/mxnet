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
using System.Diagnostics;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using MxNet.Initializers;
using MxNet.Numpy;
using MxNet.Sym.Numpy;

namespace MxNet.Gluon
{
    public abstract class Block : DynamicObject
    {
        public delegate void ApplyFn(Block block);

        public delegate void Hook(Block block, NDArrayOrSymbol input);

        public bool _active;
        public (SymbolList, _Symbol)? _cached_graph;
        public Dictionary<string, Block> _childrens;
        internal Dictionary<int, Hook> _forward_hooks;
        internal Dictionary<int, Hook> _forward_pre_hooks;
        internal Dictionary<string, Parameter> _reg_params;
        internal bool _monitor_all;
        internal Action<string, string, ndarray> _callback;
        public List<int> _in_format;
        public List<int> _out_format;

        public Block()
        {
            
            _childrens = new Dictionary<string, Block>();
            _reg_params = new Dictionary<string, Parameter>();
            _forward_hooks = new Dictionary<int, Hook>();
            _forward_pre_hooks = new Dictionary<int, Hook>();
            Params = new ParameterDict();
        }

        public virtual ParameterDict Params { get; set; }

        public object this[string name]
        {
            set
            {
                if (value is Parameter)
                    _reg_params[name] = (Parameter) value;

                if (value is Block)
                    RegisterChild((Block) value);

                if (value is HybridBlock) 
                {
                    var blk = (HybridBlock)value;
                    if (this._active)
                    {
                        Logger.Warning("Currently the model has been hybridized. Automatically deactivate the hybridization when changing the children blocks.");
                        this._active = false;
                    }

                    blk.ClearCachedOp();
                    RegisterChild(blk);
                }
            }
        }

        public Block this[int i]
        {
            get
            {
                return _childrens.Values.ToArray()[i]; ;
            }
        }

        public void SetAttr(string name, Block value)
        {
            RegisterChild(value, name);
        }

        public void SetAttr(string name, Parameter value)
        {
            if (_reg_params.ContainsKey(name))
                throw new Exception("Overriding Parameter attribute %s is not allowed. " +
                                    "If you want to share parameters between blocks, please set " +
                                    "'params' at Block construction instead.");

            _reg_params[name] = value;
        }

        public virtual Block ShareParameters(ParameterDict shared)
        {
            if (shared == null)
            {
                return this;
            }
           
            var shared_set = new HashSet<string>(shared.Keys()).ToList();
            this._shared_parameters(shared, shared_set);
            if (shared_set.Count > 0)
            {
                foreach (var name in shared_set)
                {
                    Logger.Warning($"Parameter name {name} is not in the current model!");
                }
            }
            return this;
        }

        public virtual void _shared_parameters(ParameterDict shared, List<string> shared_set, string prefix = "")
        {
            if (!string.IsNullOrWhiteSpace(prefix))
            {
                prefix += ".";
            }

            foreach (var p in this._reg_params)
            {
                var key = prefix + p.Key;
                if (shared.Get(key) != null)
                {
                    this[p.Key] = shared[key];
                    shared_set.Remove(key);
                }
            }
            foreach (var c in this._childrens)
            {
                var name = c.Key;
                var child = c.Value;
                child._shared_parameters(shared, shared_set, prefix + name);
            }
        }

        public virtual string Alias()
        {
            return GetType().Name.ToLower();
        }

        public ParameterDict CollectParams(string select = null)
        {
            return CollectParamsWithPrefix(select: select);
        }

        public virtual ParameterDict CollectParamsWithPrefix(string prefix = "", string select = null)
        {
            ParameterDict ret = new ParameterDict(); 
            if (!string.IsNullOrWhiteSpace(prefix))
            {
                prefix += ".";
            }

            if (select == null)
            {
                foreach (var item in _reg_params)
                {
                    ret.Add(prefix + item.Key, item.Value);
                }
            }
            else
            {
                var pattern = new Regex(select);
                foreach (var item in _reg_params)
                {
                    if(pattern.IsMatch(prefix + item.Key))
                        ret.Add(prefix + item.Key, item.Value);
                }
            }

            foreach (var item in this._childrens)
            {
                var name = item.Key;
                var child = item.Value;
                ret.Update(child.CollectParamsWithPrefix(prefix + name, select));
            }

            return ret;
        }

        public void SaveParameters(string filename, bool deduplicate = false)
        {
            var arg_dict = new NDArrayDict();
            var collected_params = CollectParamsWithPrefix();

            if (deduplicate)
            {
                ParameterDict reverse_params = new ParameterDict();
                foreach (var item in collected_params)
                {
                    if(!reverse_params.ContainsValue(item.Value))
                    {
                        reverse_params.Add(item.Key, item.Value);
                    }
                }

                collected_params.Clear();
                collected_params = reverse_params;
            }

            foreach (var item in collected_params.Items()) arg_dict[item.Key] = item.Value.Reduce();

            ndarray.Save(filename, arg_dict);
        }

        public void LoadParameters(string filename, Context ctx = null, bool allow_missing = false,
            bool ignore_extra = false, bool cast_dtype = false, string dtype_source = "current")
        {
            ndarray.Load(filename, out var loaded);
            LoadDict(filename, loaded, ctx, allow_missing, ignore_extra, cast_dtype, dtype_source);
        }

        public virtual void LoadDict(
                string filename,
                NDArrayDict param_dict,
                Context ctx = null,
                bool allow_missing = false,
                bool ignore_extra = false,
                bool cast_dtype = false,
                string dtype_source = "current")
        {
            var @params = this.CollectParams();
            var error_str = filename != null ? $"file: {filename}" : "param_dict";
            var loaded = param_dict.ToDictionary(_tup_1 => _tup_1.Key.StartsWith("arg:") || _tup_1.Key.StartsWith("aux:") ? _tup_1.Key.Substring(4): _tup_1.Key, _tup_1 => _tup_1.Value);

            if (!allow_missing)
            {
                var params_inv = new Dictionary<Parameter, List<string>>();
                foreach (var p in @params) {
                    if(params_inv.ContainsKey(p.Value))
                    {
                        params_inv[p.Value].Add(p.Key);
                    }
                    else
                    {
                        params_inv.Add(p.Value, new List<string>() { p.Key });
                    }
                }

                foreach (var p in @params) {
                    var name = p.Key;
                    var param = p.Value;
                    Debug.Assert((from pkey in params_inv[param]
                                     select loaded.ContainsKey(pkey)).Any(),
                                     $"Parameter '{name}' is missing in '{error_str}', which contains parameters: {string.Join(",", loaded.Keys)}. Set allow_missing=True to ignore missing parameters.");
                }
            }
            if (ctx == null)
            {
                ctx = Context.CurrentContext;
            }

            foreach (var name in loaded.Keys)
            {
                if (!ignore_extra && !@params.ContainsKey(name)) {
                    throw new Exception($"Parameter '{name}' loaded from '{error_str}' is not present in Dict, which contains parameters {string.Join(",", loaded.Keys)}. Set ignore_extra=True to ignore. ");
                }

                if (@params.ContainsKey(name)) {
                    var param = loaded[name];
                    @params[name].LoadInit(param, new Context[] { ctx }, cast_dtype: cast_dtype, dtype_source: dtype_source);
                }
            }
        }

        public virtual void RegisterChild(Block block, string name = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                name = _childrens.Count.ToString();

            _childrens[name] = block;
            //ToDo: Implement weak reference
        }

        public HookHandle RegisterForwardPreHook(Hook hook)
        {
            var handle = new HookHandle();
            handle.Attach(_forward_pre_hooks, hook);
            return handle;
        }

        public HookHandle RegisterForwardHook(Hook hook)
        {
            var handle = new HookHandle();
            handle.Attach(_forward_hooks, hook);
            return handle;
        }

        public Block Apply(ApplyFn fn)
        {
            foreach (var cld in _childrens.Values) cld.Apply(fn);

            fn(this);

            return this;
        }

        public void Initialize(Initializer init = null, Context[] ctx = null, bool verbose = false,
            bool force_reinit = false)
        {
            init = init ?? new Uniform();
            CollectParams().Initialize(init, ctx, verbose, force_reinit);
        }

        private int _save_cached_graphs(Block blk, Dictionary<string, object> structure, int index)
        {
            // create new entry for this block
            var mdl = new Dictionary<string, object>();

            // encode unique name based on block type and ID
            var name = blk.GetType().Name.ToLower();
            structure[name + index.ToString()] = mdl;
            index += 1;
            if (blk is HybridBlock)
            {
                if (blk._cached_graph != null)
                {
                    // save in/out formats
                    mdl["in_format"] = blk._in_format;
                    mdl["out_format"] = blk._out_format;
                    // save cached graph & input symbols
                    var _tup_1 = blk._cached_graph.Value;
                    var syms = _tup_1.Item1;
                    var @out = _tup_1.Item2;
                    var mdl_syms = new List<string>();
                    foreach (var sym in syms)
                    {
                        mdl_syms.Add(sym.ToJSON());
                    }

                    mdl["inputs"] = mdl_syms;
                    mdl["symbol"] = @out.ToJSON();
                    mdl["hybridized"] = true;
                }
                else
                {
                    mdl["hybridized"] = false;
                }
            }
            // save param uuids
            var pmap = new Dictionary<string, string>();
            var pnames = blk.Params.Keys();

            foreach (var p in pnames)
            {
                var param = blk.Params[p];
                pmap[p] = param._uuid;
            }

            mdl["params"] = pmap;

            // recursively save children
            foreach (var child in blk._childrens.Values)
            {
                index = _save_cached_graphs(child, mdl, index);
            }

            // return latest index (ie. block count)
            return index;
        }

        public virtual void Save(string prefix)
        {
            // create empty model structure
            var model = new Dictionary<string, object>();

            // save top-level block
            _save_cached_graphs(this, model, 0);
            var json = Newtonsoft.Json.JsonConvert.SerializeObject(model);
            File.WriteAllText(prefix + "-model.json", json);
            // save params
            this.SaveParameters("MyModel-model.params");
        }

        private int _load_cached_graphs(Block blk, Dictionary<string, object> structure, int index)
        {
            var name = this.Alias();
            // lookup previous encoded name based on block type and ID
            var mdl = (Dictionary<string, object>)structure[name + index.ToString()];
            index += 1;
            if (blk is HybridBlock)
            {
                if (mdl.ContainsKey("hybridized"))
                {
                    // restore in/out formats
                    blk._in_format = (List<int>)mdl["in_format"];
                    blk._out_format = (List<int>)mdl["out_format"];
                    // get saved symbol
                    var @out = _Symbol.FromJSON(mdl["symbol"].ToString());
                    var syms = new List<_Symbol>();
                    // recreate inputs for this symbol
                    foreach (var inp in (string[])mdl["inputs"])
                    {
                        syms.Add(_Symbol.FromJSON(inp));
                    }

                    // reset cached_graph and active status
                    blk._cached_graph = (syms, @out);
                    blk._active = true;
                }
            }
            // reload param uuids
            var pmap = (Dictionary<string, string>)mdl["params"];
            foreach (var p in pmap)
            {
                var param = blk.Params[p.Key];
                param._uuid = p.Value;
            }

            // recursively reload children
            foreach (var child in blk._childrens)
            {
                index = _load_cached_graphs(child.Value, mdl, index);
            }
            // return latest index (ie. block count)
            return index;
        }

        public virtual void Load(string prefix)
        {
            // load model json from file
            var json = File.ReadAllText(prefix + "-model.json");
            var model = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(json);

            // load top-level block
            _load_cached_graphs(this, model, 0);
            // load params
            this.LoadParameters("MyModel-model.params");
        }

        public virtual void Hybridize(bool active = true, bool partition_if_dynamic = true, bool static_alloc = false, bool static_shape = false,
            int inline_limit = 2, int? forward_bulk_size = null, int? backward_bulk_size = null)
        {
            foreach (var cld in _childrens.Values) cld.Hybridize(active, static_alloc, static_shape);
        }

        public virtual void Cast(DType dtype)
        {
            foreach (var item in _childrens.Values) item.Cast(dtype);

            foreach (var item in Params.Items()) item.Value.Cast(dtype);
        }

        public virtual void zero_grad()
        {
            // collect gradient arrays for each ctx
            var arrays = new Dictionary<Context, NDArrayList>();
            var @params = this.CollectParams();
            foreach (var p in @params.Values()) {
                if (p.GradReg ==  OpGradReq.Null || p._grad == null)
                {
                    continue;
                }

                var grads = p.ListGrad();
                for(int i = 0; i< grads.Length; i++)
                {
                    var g = grads[i];
                    if (g.stype == StorageStype.RowSparse)
                    {
                        g = nd.ZerosLike(g);
                    }
                    else
                    {
                        if (arrays.ContainsKey(g.ctx))
                            arrays.Add(g.ctx, g);
                        else
                            arrays[g.ctx].Add(g);
                    }
                }
            }
            if (arrays.Count == 0)
            {
                return;
            }

            foreach (var arr in arrays.Values)
            {
                nd.ResetArrays(arr);
            }
        }

        public virtual void ResetCtx(Context ctx)
        {
            var @params = this.CollectParams();
            foreach (var i in @params.Values()) {
                i.ResetCtx(ctx);
            }
        }

        public virtual NDArrayOrSymbolList Call(NDArrayOrSymbolList inputs)
        {
            foreach (var hook in _forward_pre_hooks.Values) hook(this, inputs[0]);

            var @out = Forward(inputs);
            foreach (var hook in _forward_hooks.Values) hook(this, @out);

            return @out;
        }

        public abstract NDArrayOrSymbolList Forward(NDArrayOrSymbolList inputs);

        public virtual void RegisterOpHook(Action<string, string, ndarray> callback, bool monitor_all = false)
        {
            foreach (var cld in this._childrens.Values)
            {
                cld.RegisterOpHook(callback, monitor_all);
            }
        }

        public virtual void Summary(NDArrayList inputs)
        {
            //ToDo: Implement Summmary
            foreach (var item in inputs)
            {
            }
        }

        internal static (NDArrayOrSymbolList, int[]) Flatten(NDArrayOrSymbolList args, string inout_str)
        {
            var flat = new List<NDArrayOrSymbol>();
            var fmts = new List<int>();
            foreach (var arg in args)
                if (arg.IsNDArray)
                {
                    flat.Add(arg.NdX);
                    fmts.Add(1);
                }
                else if (arg.IsSymbol)
                {
                    var len = arg.SymX.ListOutputs().Count;
                    flat.Add(arg.SymX);
                    fmts.Add(len);
                }

            return (flat.ToArray(), fmts.ToArray());
        }

        internal static (NDArrayOrSymbol[], NDArrayOrSymbol[]) Regroup(List<NDArrayOrSymbol[]> args, List<int> fmt)
        {
            var ret = new List<NDArrayOrSymbol>();
            var args_ret = new List<NDArrayOrSymbol>();

            foreach (var i in fmt)
            {
                if (i == 0)
                {
                    ret.AddRange(args[0]);
                    foreach (var item in args.Skip(1)) args_ret.AddRange(item);

                    continue;
                }

                for (var j = 0; j < i; j++)
                    ret.AddRange(args[j]);

                for (var j = i; j < args.Count; j++)
                    args_ret.AddRange(args[j]);
            }

            return (ret.ToArray(), args_ret.ToArray());
        }

        internal static string CommonPrefix(string[] names)
        {
            if (names == null)
                return "";

            var prefix = names[0];
            foreach (var name in names)
            {
                var i = 0;
                while (i < prefix.Length && i < name.Length && prefix[i] == name[i])
                    i++;

                prefix = prefix.Substring(0, i);
            }

            return prefix;
        }

        internal static (DType[], DType[]) InferParamTypes(SymbolList in_params, _Symbol out_params, string[] arg_params,
            string[] aux_params, DType default_dtype = null)
        {
            DType[] arg_types = null;
            DType[] aux_types = null;
            DType[] _ = null;

            var input_sym_names = in_params.Select(x => x.Name).ToArray();
            var input_sym_arg_types = new List<DType>();
            var can_infer_input_type = true;
            foreach (var in_param in in_params)
            {
                var input_sym_arg_type = in_param.InferType().Item1;
                if (input_sym_arg_type == null || input_sym_arg_type.Length < 1)
                {
                    can_infer_input_type = false;
                    break;
                }

                input_sym_arg_types.Add(input_sym_arg_type[0]);
            }

            if (can_infer_input_type)
            {
                var @params = new Dictionary<string, DType>();
                var i = 0;
                foreach (var k in input_sym_names)
                {
                    @params.Add(k, input_sym_arg_types[i]);
                    i++;
                }

                try
                {
                    (arg_types, _, aux_types) = out_params.InferType(@params);
                }
                catch (MXNetException ex)
                {
                    arg_types = null;
                    aux_types = null;
                }
            }

            if (arg_types == null || arg_types.Length != arg_params.Length)
            {
                arg_types = new DType[arg_params.Length];
                for (var i = 0; i < arg_params.Length; i++) arg_types[i] = default_dtype;
            }

            if (aux_types == null || aux_types.Length != arg_params.Length)
            {
                aux_types = new DType[arg_params.Length];
                for (var i = 0; i < arg_params.Length; i++) aux_types[i] = default_dtype;
            }

            return (arg_types, aux_types);
        }

        public static (bool, bool, Context[], Context) GatherTypeCtxInfo(NDArrayOrSymbolList args)
        {
            var has_symbol = false;
            var has_ndarray = false;
            var ctx_set = new List<Context>();
            Context first_ctx = null;
            foreach (var ele in args)
            {
                var _tup_1 = GatherTypeCtxInfo(ele);
                var ele_has_sym = _tup_1.Item1;
                var ele_has_nd = _tup_1.Item2;
                var ele_ctx_set = _tup_1.Item3;
                var ele_first_ctx = _tup_1.Item4;
                has_symbol = has_symbol || ele_has_sym;
                has_ndarray = has_ndarray || ele_has_nd;
                if (first_ctx == null && ele_first_ctx != null)
                {
                    first_ctx = ele_first_ctx;
                }

                foreach (var item in ele_ctx_set)
                {
                    if (!ctx_set.Contains(item))
                        ctx_set.Add(item);
                }
                
                if (has_symbol && has_ndarray)
                {
                    break;
                }
            }

            return (has_symbol, has_ndarray, ctx_set.ToArray(), first_ctx);
        }

        public static (bool, bool, Context[], Context) GatherTypeCtxInfo(NDArrayOrSymbol args)
        {
            if (args.IsNDArray)
            {
                return(false, true, new Context[]{ args.NdX.ctx }, args.NdX.ctx);
            }
            else if (args.IsSymbol)
            {
                return (true, false, new Context[0], null);
            }
            else
            {
                return (false, false, null, null);
            }
        }

        public override string ToString()
        {
            var modstr = string.Join("\n", _childrens.Select(i => $"  ({i.Key}): {Utils.Indent(i.Value.ToString(), 2)}"));
            return $"{GetType().Name}(\n{modstr}\n)";
        }
    }
}