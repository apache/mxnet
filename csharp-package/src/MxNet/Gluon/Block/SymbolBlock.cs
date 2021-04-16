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
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MxNet.Gluon
{
    public class SymbolBlock : HybridBlock
    {
        public SymbolBlock(SymbolList outputs, SymbolList inputs, ParameterDict @params = null)
            : base()
        {
            Construct(outputs, inputs, @params);
        }

        public void Construct(SymbolList outputs, SymbolList inputs, ParameterDict @params = null)
        {
            SymbolList syms = null;
            _Symbol @out = null;

            var (s, _in_format) = Flatten(inputs, "input");
            var (o, _out_format) = Flatten(outputs, "output");
            syms = s.ToSymbols();
            @out = _Symbol.Group(o.ToSymbols());

            List<string> input_names = new List<string>();
            foreach (var item in syms)
            {
                if (item.ListOutputs().Count != 1)
                    throw new Exception($"Input symbols must be variable, but {item.Name} is an output of operators");

                if (input_names.Contains(item.Name))
                    continue;

                input_names.Add(item.Name);
            }

            //ToDo: check if any symbol is row_sparse

            var arg_params = @out.ListArguments().ToArray();
            var aux_params = @out.ListAuxiliaryStates().ToArray();
            var (arg_types, aux_types) = InferParamTypes(syms, @out, arg_params, aux_params);
            if (@params == null) {
                @params = new ParameterDict();
            }

            var unused_params = new HashSet<string>(@params.Keys()).ToList();
            foreach (var item in new HashSet<string>(arg_params).ToList())
            {
                if (unused_params.Contains(item))
                    unused_params.Remove(item);
            }

            foreach (var item in new HashSet<string>(aux_params).ToList())
            {
                if (unused_params.Contains(item))
                    unused_params.Remove(item);
            }

            if (unused_params.Count > 0)
            {
                throw new Exception($"{string.Join(",", unused_params)} params are unused by the model.");
            }

            this._reg_params = @params;
            for (int i = 0; i < arg_params.Length; i++)
            {
                var arg = arg_params[i];
                if (!input_names.Contains(arg))
                {
                    Params.Get(arg, OpGradReq.Null, allow_deferred_init: true, dtype: arg_types[i]);
                }
            }

            _cached_graph = (syms, @out);
            int len_prefix = CommonPrefix(Params.Keys()).Length;
            _reg_params = new ParameterDict();
            foreach (var item in Params)
            {
                _reg_params.Add(item.Key.Remove(0, len_prefix), item.Value);
            }
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            return args;
        }

        public override NDArrayOrSymbolList Forward(NDArrayOrSymbolList inputs)
        {
            if (DeferredCompute.IsDeferredCompute())
            {
                throw new Exception("Calling a SymbolBlock from within HybridBlock is not yet supported in Gluon 2.");
            }

            if (inputs[0].IsNDArray)
            {
                return CallCachedOp(inputs);
            }

            var (args, in_fmt) = Flatten(inputs, "input");
            if (in_fmt != _in_format.ToArray())
                throw new Exception("Invalid input format");

            var ret = _cached_graph.Value.Item2.ShallowCopy();
            SymbolDict composeArgs = new SymbolDict();
            for(int i = 0;i< _cached_graph.Value.Item1.Length;i++)
            {
                composeArgs.Add(_cached_graph.Value.Item1[0].Name, args[i]);
            }

            ret.Compose(composeArgs);

            return Regroup(new List<NDArrayOrSymbolList>() { new NDArrayOrSymbolList { ret } }, _out_format.ToList()).Item1[0];
        }

        public static SymbolBlock Imports(string symbol_file, string[] input_names, string param_file = null,
            Context[] ctx = null)
        {
            _Symbol sym = _Symbol.Load(symbol_file);
            SymbolList inputs = null;
            if (string.IsNullOrWhiteSpace(param_file))
                inputs = input_names.Select(x => (_Symbol.Var(x, dtype: DType.Float32))).ToArray();
            else
                inputs = input_names.Select(x => (_Symbol.Var(x))).ToArray();

            var ret = new SymbolBlock(new SymbolList { sym }, inputs);

            if(!string.IsNullOrWhiteSpace(param_file))
            {
                var p = ret.CollectParams();
                p.Load(param_file, ctx: ctx, cast_dtype: true, dtype_source: "saved");
            }

            return ret;
        }

        public override void ClearCachedOp()
        {
            var tmp = _cached_graph;
            base.ClearCachedOp();
            _cached_graph = tmp;
        }

        public override void Cast(DType dtype)
        {
            ClearCachedOp();
            base.Cast(dtype);
            //ToDo support for float 16
            //if (np.dtype(dtype).name == "float16")
            //{
            //    // correct BatchNorm types back to float32 due to its special requirement
            //    var @out = this._cached_graph[1];
            //    var params_list = @out.get_internals().list_inputs();
            //    foreach (var node in params_list)
            //    {
            //        if (node.endswith("running_var"))
            //        {
            //            prefix = node[:: - 11];
            //            sibs = (from t in ("running_mean", "gamma", "beta")
            //                    select (prefix + t)).ToList();
            //            is_bn = all(from p in sibs
            //                        select params_list.Contains(p));
            //            if (is_bn)
            //            {
            //                this.params.get(node).cast("float32");
            //                foreach (var sib in sibs)
            //                {
            //                    this.params.get(sib).cast("float32");
            //                }
            //            }
            //        }
            //        if (node.endswith("moving_var"))
            //        {
            //            // another convention used
            //            prefix = node[:: - 10];
            //            sibs = (from t in ("moving_mean", "gamma", "beta")
            //                    select (prefix + t)).ToList();
            //            is_bn = all(from p in sibs
            //                        select params_list.Contains(p));
            //            if (is_bn)
            //            {
            //                this.params.get(node).cast("float32");
            //                foreach (var sib in sibs)
            //                {
            //                    this.params.get(sib).cast("float32");
            //                }
            //            }
            //        }
            //    }
            //}
        }
    }
}