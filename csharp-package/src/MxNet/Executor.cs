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
using MxNet.Interop;
using mx_uint = System.UInt32;
using ExecutorHandle = System.IntPtr;
using MxNet.IO;
using MxNet.Sym.Numpy;

// ReSharper disable once CheckNamespace
namespace MxNet
{
    public sealed class Executor : DisposableMXNetObject
    {
        #region Fields

        public string[] _arg_names;

        public NDArrayList _args;

        public NDArrayDict _args_grad;

        public string[] _aux_names;

        public CachedOp _cached_op;

        public Context _ctx;

        public Dictionary<string, OpGradReq> _grad_req;

        public string[] _input_names;

        public string[] _output_names;

        public bool _requires_grad;

        public NDArrayList outputs;

        #endregion

        #region Constructors

        public Executor(Symbol symbol,
            Context context,
            NDArrayDict argmentArrays,
            NDArrayDict gradientArrays,
            Dictionary<string, OpGradReq> gradReqs,
            NDArrayList auxiliaryArrays)
            : this(symbol, context, argmentArrays, gradientArrays, gradReqs, auxiliaryArrays,
                new NDArrayDict(), null)
        {
        }

        public Executor(Symbol symbol,
            Context context,
            NDArrayDict argmentArrays,
            NDArrayDict gradientArrays,
            Dictionary<string, OpGradReq> gradReqs,
            NDArrayList auxiliaryArrays,
            NDArrayDict aux_states)
            : this(symbol, context, argmentArrays, gradientArrays, gradReqs, auxiliaryArrays, aux_states, null)
        {
        }

        public Executor(Symbol sym,
            Context ctx,
            NDArrayDict args,
            NDArrayDict args_grad,
            Dictionary<string, OpGradReq> grad_req,
            NDArrayList auxiliaryArrays,
            NDArrayDict aux_states,
            Executor sharedExec)
        {
            this.outputs = null;
            this._input_names = sym.ListInputs().ToArray();
            this._aux_names = sym.ListAuxiliaryStates().ToArray();
            this._arg_names = sym.ListArguments().ToArray();
            this._output_names = sym.ListOutputs().ToArray();
            this._ctx = ctx;
            this._grad_req = grad_req;
            // grad_req
            this._requires_grad = false;
            foreach (var x in grad_req)
            {
                if (this._input_names.Contains(x.Key) && x.Value != OpGradReq.Null)
                {
                    this._requires_grad = true;
                }
            }

            // args grad
            this._args_grad = args_grad;

            // args
            this._args = new NDArrayList(this._input_names.Length);
            foreach (var x in args)
            {
                try
                {
                    var i = this._input_names.ToList().IndexOf(x.Key);
                    this._args[i] = x.Value.ChangeContext(ctx);
                }
                catch (Exception)
                {
                    // ignore provided arg which is not present in
                    // input_names
                }
            }

            // aux states
            if (aux_states != null)
            {
                foreach (var x in aux_states)
                {
                    if (this._aux_names.Contains(x.Key))
                    {
                        var i = this._input_names.ToList().IndexOf(x.Key);
                        this._args[i] = x.Value.ChangeContext(ctx);
                    }
                }
            }
            // arg grad
            if (this._args_grad != null)
            {
                foreach (var x in this._args_grad)
                {
                    try
                    {
                        var i = this._input_names.ToList().IndexOf(x.Key);
                        var req = grad_req[x.Key];

                        if (req !=  OpGradReq.Null)
                        {
                            this._args[i].AttachGrad(req, stype: x.Value.stype);
                             x.Value.CopyTo(this._args[i].grad);
                        }
                    }
                    catch (Exception)
                    {
                        // ignore provided arg which is not present in
                        // input_names
                    }
                }
            }

            this._cached_op = new CachedOp(sym);
        }

        #endregion

        #region Properties

        internal ExecutorHandle Handle { get; }

        public NDArrayList Outputs { get; }

        public NDArray Output => Outputs.First();

        public NDArrayList ArgmentArrays { get; internal set; }

        public NDArrayList GradientArrays { get; internal set; }

        public NDArrayList AuxiliaryArrays { get; internal set; }

        #endregion

        #region Methods

        public Symbol GetOptimizedSymbol()
        {
            return this._cached_op.GetOptimizedSymbol();
        }

        public NDArrayDict ArgmentDictionary()
        {
            return GetDictionary(GetOptimizedSymbol().ListArguments(), ArgmentArrays);
        }

        public NDArrayDict GradientDictionary()
        {
            return GetDictionary(GetOptimizedSymbol().ListArguments(), GradientArrays);
        }

        public NDArrayDict AuxiliaryDictionary()
        {
            return GetDictionary(GetOptimizedSymbol().ListAuxiliaryStates(), AuxiliaryArrays);
        }

        public NDArrayDict OutputDictionary()
        {
            return GetDictionary(GetOptimizedSymbol().ListOutputs(), Outputs);
        }

        public void Backward()
        {
            Backward(new NDArrayList());
        }

        public void Backward(NDArrayList out_grads = null)
        {
            if (out_grads != null)
            {
                out_grads = (from o in out_grads
                             select o.ChangeContext(this._ctx)).ToList();
            }

            if (this._requires_grad)
            {
                if (this.outputs == null)
                {
                    this.Forward();
                }

                Autograd.Backward(this.outputs, head_grads: out_grads);

                foreach (var x in this._args_grad)
                {
                    var k = x.Key;
                    var v = x.Value;
                    try
                    {
                        var i = this._input_names.ToList().IndexOf(k);
                        if (this._args[i].grad != null)
                        {
                            v = this._args[i].grad;
                        }
                    }
                    catch (Exception)
                    {
                        // ignore provided arg grad which is not present in
                        // input_names
                    }
                }
            }
        }

        public NDArrayList Forward(bool isTrain = false, NDArrayDict kwargs = null)
        {
            if (kwargs != null)
            {
                foreach (var x in kwargs)
                {
                    var name = x.Key;
                    var array = x.Value;
                    if (this._input_names.Contains(name))
                    {
                        var index = this._input_names.ToList().IndexOf(name);
                        if (this._args.Length < index)
                        {
                            this._args.Add(array);
                            var req = _grad_req[name];
                            if (req != OpGradReq.Null)
                            {
                                this._args[index].AttachGrad(req);
                            }
                        }
                        else
                        {
                            this._args[index] = array;
                        }
                    }
                }
            }

            using(var ag = Autograd.Record(train_mode: isTrain)) { 
                this.outputs = this._cached_op.Call(this._args);
            }
            
            return this.outputs;
        }

        public void CopyFromParams(NDArrayDict arg_params, NDArrayDict aux_params = null,
            bool allow_extra_params = false)
        {
            var arg_dict = ArgmentDictionary();
            var aux_dict = AuxiliaryDictionary();
            foreach (var item in arg_params)
                if (arg_dict.Contains(item.Key))
                {
                    var dst = arg_dict[item.Key];
                    //ToDo: Missing AMP cast for float16
                    item.Value.AsType(dst.dtype).CopyTo(dst);
                }
                else if (!allow_extra_params)
                {
                    throw new Exception($"Find name \"{item.Key}\" that is not in the arguments");
                }

            if (aux_params == null)
                return;

            foreach (var item in aux_params)
                if (aux_dict.Contains(item.Key))
                {
                    var dst = aux_dict[item.Key];
                    //ToDo: Missing AMP cast for float16
                    item.Value.AsType(dst.dtype).CopyTo(dst);
                }
                else if (!allow_extra_params)
                {
                    throw new Exception($"Find name \"{item.Key}\" that is not in the auxiliary states");
                }
        }

        #region Overrids

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            NativeMethods.MXExecutorFree(Handle);
        }

        #endregion

        #region Helpers

        private static NDArrayDict GetDictionary(IList<string> names, NDArrayList arrays)
        {
            var ret = new NDArrayDict();

            var set = new HashSet<string>();
            foreach (var s in names)
            {
                Logging.CHECK(!set.Contains(s), $"Duplicate names detected, {s}");
                set.Add(s);
            }

            Logging.CHECK_EQ(set.Count, arrays.Length, "names size not equal to arrays size");
            for (var i = 0; i < names.Count; ++i)
                ret[names[i]] = arrays[i];

            return ret;
        }

        #endregion

        #endregion
    }
}