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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MxNet.Initializers;
using MxNet.Numpy;

namespace MxNet.Gluon
{
    public class ParameterDict : IEnumerable<KeyValuePair<string, Parameter>>
    {
        private readonly Dictionary<string, Parameter> _params;

        public ParameterDict(string prefix = "", ParameterDict shared = null)
        {
            Prefix = prefix;
            Shared = shared;
            _params = new Dictionary<string, Parameter>();
        }

        public string Prefix { get; }

        public ParameterDict Shared { get; }

        public Parameter this[string name]
        {
            get
            {
                if (_params.ContainsKey(name))
                    return _params[name];

                string key = _params.Keys.Where(x => x.Contains(name)).FirstOrDefault();
                if (!string.IsNullOrWhiteSpace(key))
                    return _params[key];

                return null;
            }
            set
            {
                if (_params.ContainsKey(name))
                    _params[name] = value;
                else
                    _params.Add(name, value);

            }
        }

        public string[] Keys()
        {
            return _params.Keys.ToArray();
        }

        public Parameter[] Values()
        {
            return _params.Values.ToArray();
        }

        public Dictionary<string, Parameter> Items()
        {
            return _params;
        }

        public bool Contains(string key)
        {
            return _params.ContainsKey(key);
        }

        public bool Contains(Parameter value)
        {
            return _params.Values.Where(x => x.Name == value.Name).Count() > 0;
        }

        public void Add(string name, Parameter value)
        {
            _params.Add(name, value);
        }

        public void Clear()
        {
            _params.Clear();
        }

        public Parameter GetConstant(string name, ndarray value = null)
        {
            name = Prefix + name;
            var param = GetImpl(name);
            if (param == null)
            {
                if (value == null)
                    throw new Exception($"No constant named '{name}'. Please specify value " +
                                        "if you want to create a new constant.");

                param = new Constant(name, value);
                _params[name] = param;
            }
            else if (value != null)
            {
                if (!(param is Constant))
                    throw new Exception($"Parameter '{name}' already exists but it is not a constant.");
            }

            return param;
        }


        private Parameter GetImpl(string name)
        {
            if (_params.ContainsKey(name))
                return _params[name];

            if (Shared != null && Shared.Contains(name))
                _params[name] = Shared[name];

            return null;
        }

        public Parameter Get(string name, OpGradReq grad_req = OpGradReq.Write, Shape shape = null, DType dtype = null,
            float lr_mult = 1.0f, float wd_mult = 1.0f, Initializer init = null, bool allow_deferred_init = false,
            bool differentiable = true, StorageStype stype = StorageStype.Default,
            StorageStype grad_stype = StorageStype.Default)
        {
            name = Prefix + name;
            var param = GetImpl(name);
            if (param == null)
            {
                param = new Parameter(name, grad_req, shape, dtype, lr_mult, wd_mult, init, allow_deferred_init,
                    differentiable, stype, grad_stype);
                _params[name] = param;
            }
            else
            {
                param._shape = param.Shape ?? shape;
                param.Init = param.Init ?? init;
            }

            return param;
        }

        public void Update(ParameterDict other)
        {
            foreach (var item in other.Items())
            {
                if (!_params.ContainsKey(item.Key))
                {
                    _params.Add(item.Key, item.Value);
                    continue;
                }

                if (_params[item.Key].GetType() == item.Value.GetType())
                    _params[item.Key] = item.Value;
                else
                    throw new Exception("Cannot update self with other because they have different " +
                                        $"Parameters with the same name '{item.Key}'");
            }
        }

        public void Initialize(Initializer init = null, Context[] ctx = null, bool verbose = false,
            bool force_reinit = false)
        {
            init = init ?? new Uniform();
            if (verbose)
                init.SetVerbosity(verbose);

            var keys = _params.Keys.ToList();
            foreach (var p in _params)
            {
                p.Value.Initialize(null, ctx, init, force_reinit);
            }
        }


        public void ZeroGrad()
        {
            foreach (var item in _params) item.Value.ZeroGrad();
        }

        public void ResetCtx(Context ctx)
        {
            foreach (var item in _params) item.Value.ZeroGrad();
        }

        public void Save(string filename, string strip_prefix = "")
        {
            var args_dict = new NDArrayDict();
            foreach (var param in _params)
            {
                if (strip_prefix != "" && !param.Key.StartsWith(strip_prefix))
                    throw new Exception($"Prefix '{strip_prefix}' is to be striped before saving, but Parameter's " +
                                        $"name '{param.Key}' does not start with '{strip_prefix}'. " +
                                        "this may be due to your Block shares parameters from other " +
                                        "Blocks or you forgot to use 'with name_scope()' when creating " +
                                        "child blocks. For more info on naming, please see " +
                                        "http://mxnet.incubator.apache.org/tutorials/basic/naming.html");

                args_dict[param.Key.Remove(0, strip_prefix.Length)] = param.Value.Reduce();
            }

            ndarray.Save(filename, args_dict);
        }

        public void Load(string filename, Context[] ctx = null, bool allow_missing = false,
            bool ignore_extra = false, string restore_prefix = "", bool cast_dtype = false,
            string dtype_source = "current")
        {
            if (!string.IsNullOrWhiteSpace(restore_prefix))
                foreach (var name in Keys())
                    if (!name.StartsWith(restore_prefix))
                        throw new Exception(
                            $"restore_prefix is '{restore_prefix}' but Parameters name '{name}' does not start with '{restore_prefix}'");

            var lprefix = restore_prefix.Length;
            var loaded_ndarray = ndarray.Load(filename);
            var arg_dict = new NDArrayDict();
            foreach (var item in loaded_ndarray)
            {
                var key = item.Key.StartsWith("arg:") || item.Key.StartsWith("aux:") ? item.Key.Remove(0, 4) : item.Key;
                key = restore_prefix + key;
                arg_dict[key] = item.Value;
            }

            if (!allow_missing)
                foreach (var name in Keys())
                    if (!arg_dict.Contains(name))
                        throw new Exception(
                            $"Parameter '{name.Remove(0, lprefix)}' is missing in file '{filename}', which contains parameters: {Utils.BriefPrintList(Keys().ToList())}. " +
                            "Please make sure source and target networks have the same prefix.");

            foreach (var name in arg_dict.Keys)
            {
                if (!_params.ContainsKey(name))
                {
                    if (ignore_extra)
                        throw new Exception(
                            $"Parameter '{name.Remove(0, lprefix)}' loaded from file '{filename}' is not present in ParameterDict, " +
                            $"choices are: {Utils.BriefPrintList(Keys().ToList())}. Set ignore_extra to True to ignore. " +
                            "Please make sure source and target networks have the same prefix.");

                    continue;
                }

                this[name].LoadInit(arg_dict[name], ctx, cast_dtype, dtype_source);
            }
        }

        public IEnumerator<KeyValuePair<string, Parameter>> GetEnumerator()
        {
            return _params.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}