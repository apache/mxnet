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
using MxNet.Numpy;
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace MxNet.Gluon.RNN
{
    public class StateInfo
    {
        public StateInfo()
        {
        }

        public StateInfo(FuncArgs args)
        {
            foreach (var arg in args)
            {
                if (arg.Value == null)
                    continue;

                switch (arg.Key.ToLower())
                {
                    case "shape":
                        Shape = (Shape) arg.Value;
                        break;
                    case "layout":
                        Layout = arg.Value.ToString();
                        break;
                    case "in_layout":
                        Layout = arg.Value.ToString();
                        break;
                    case "mean":
                        Mean = Convert.ToSingle(arg.Value);
                        break;
                    case "std":
                        Mean = Convert.ToSingle(arg.Value);
                        break;
                    case "dtype":
                        DataType = (DType) arg.Value;
                        break;
                    case "ctx":
                        Ctx = (Context) arg.Value;
                        break;
                }
            }
        }

        public Shape Shape { get; set; }

        public string Layout { get; set; }

        public float Mean { get; set; }

        public float Std { get; set; }

        public DType DataType { get; set; }

        public Context Ctx { get; set; }

        public string Name { get; set; }

        public void Update(FuncArgs args)
        {
            foreach (var arg in args)
            {
                if (arg.Value == null)
                    continue;

                switch (arg.Key.ToLower())
                {
                    case "shape":
                        Shape = (Shape) arg.Value;
                        break;
                    case "layout":
                        Layout = arg.Value.ToString();
                        break;
                    case "in_layout":
                        Layout = arg.Value.ToString();
                        break;
                    case "mean":
                        Mean = Convert.ToSingle(arg.Value);
                        break;
                    case "std":
                        Std = Convert.ToSingle(arg.Value);
                        break;
                    case "dtype":
                        DataType = (DType) arg.Value;
                        break;
                    case "ctx":
                        Ctx = (Context) arg.Value;
                        break;
                    case "name":
                        Name = arg.Value.ToString();
                        break;
                }
            }
        }

        public object[] GetArgs(string[] keys)
        {
            var args = new List<object>();
            foreach (var item in keys)
                switch (item.ToLower())
                {
                    case "shape":
                        args.Add(Shape);
                        break;
                    case "layout":
                        args.Add(Layout);
                        break;
                    case "in_layout":
                        args.Add(Layout);
                        break;
                    case "mean":
                        args.Add(Mean);
                        break;
                    case "std":
                        args.Add(Std);
                        break;
                    case "dtype":
                        args.Add(DataType);
                        break;
                    case "ctx":
                        args.Add(Ctx);
                        break;
                    case "name":
                        args.Add(Name);
                        break;
                    default:
                        args.Add(null);
                        break;
                }

            return args.ToArray();
        }
    }

    public abstract class RecurrentCell : Block
    {
        internal new Dictionary<string, RecurrentCell> _childrens = new Dictionary<string, RecurrentCell>();
        internal int _counter;
        internal int _init_counter;
        internal bool _modified;

        public RecurrentCell() : base()
        {
            _modified = false;
            Reset();
        }

        public virtual void Reset()
        {
            _init_counter = 0;
            _counter = 0;
            foreach (var cell in _childrens.Values) cell.Reset();
        }

        public abstract StateInfo[] StateInfo(int batch_size = 0);

        public override NDArrayOrSymbolList Forward(NDArrayOrSymbolList inputs)
        {
            _counter++;
            return inputs;
        }

        public virtual NDArrayOrSymbolList BeginState(int batch_size = 0, string func = null, FuncArgs args = null)
        {
            if (_modified)
                throw new Exception("After applying modifier cells (e.g. ZoneoutCell) the base " +
                                    "cell cannot be called directly. Call the modifier cell instead.");

            var states = new NDArrayOrSymbolList();
            var state_info = StateInfo(batch_size);
            for (var i = 0; i < state_info.Length; i++)
            {
                var info = state_info[i];
                _init_counter++;
                args.Add("name", $"{Alias()}begin_state_{_init_counter}");
                if (info != null)
                    info.Update(args);
                else
                    info = new StateInfo(args);

                if (func.StartsWith("sym."))
                {
                    var obj = new sym();
                    var m = typeof(sym).GetMethod(func.Replace("sym.", ""), BindingFlags.Static);
                    var keys = m.GetParameters().Select(x => x.Name).ToArray();
                    var paramArgs = info.GetArgs(keys);
                    states.Add((_Symbol) m.Invoke(obj, paramArgs));
                }
                else if (func.StartsWith("nd."))
                {
                    var obj = new nd();
                    var m = typeof(sym).GetMethod(func.Replace("nd.", ""), BindingFlags.Static);
                    var keys = m.GetParameters().Select(ids => ids.Name).ToArray();
                    var paramArgs = info.GetArgs(keys);
                    states.Add((ndarray) m.Invoke(obj, paramArgs));
                }
            }

            return states;
        }

        public virtual (NDArrayOrSymbolList, NDArrayOrSymbolList) Unroll(int length, NDArrayOrSymbolList inputs,
            NDArrayOrSymbolList begin_state = null,
            string layout = "NTC", bool? merge_outputs = null, _Symbol valid_length = null)
        {
            if (!inputs[0].IsSymbol)
                throw new Exception("Only symbols is supported");

            Reset();
            var (f_inputs, axis, batch_size) = RNNCell.FormatSequence(length, inputs, layout, false);
            begin_state = RNNCell.GetBeginState(this, begin_state, f_inputs, batch_size);
            var states = begin_state;
            var outputs = new NDArrayOrSymbolList();
            var all_states = new List<NDArrayOrSymbolList>();
            for (var i = 0; i < length; i++)
            {
                var (output, u_states) = Unroll(1, inputs[i], states);
                outputs.Add(output[0]);
                if (valid_length != null)
                    all_states.Add(u_states);
            }

            if (valid_length != null)
            {
                states = new NDArrayOrSymbolList(all_states
                    .Select(ele_list => sym.SequenceLast(sym.Stack(ele_list.ToSymbols(), ele_list.Length),
                        valid_length, true)).ToArray());

                outputs = RNNCell.MaskSequenceVariableLength(outputs, length, valid_length, axis, true);
            }

            outputs = RNNCell.FormatSequence(length, outputs, layout, merge_outputs.Value).Item1;
            return (outputs, states);
        }

        internal _Symbol Activation(_Symbol input, string activation, FuncArgs args = null, string name = "")
        {
            switch (activation.ToLower())
            {
                case "tanh":
                    return sym.Activation(input, ActivationType.Tanh, name);
                case "relu":
                    return sym.Activation(input, ActivationType.Relu, name);
                case "sigmoid":
                    return sym.Activation(input, ActivationType.Sigmoid, name);
                case "softrelu":
                    return sym.Activation(input, ActivationType.Softrelu, name);
                case "softsign":
                    return sym.Activation(input, ActivationType.Softsign, name);
                case "leakyrely":
                    return sym.LeakyReLU(input);
            }

            return input;
        }

        internal ndarray Activation(ndarray input, string activation, FuncArgs args = null)
        {
            switch (activation.ToLower())
            {
                case "tanh":
                    return nd.Activation(input, ActivationType.Tanh);
                case "relu":
                    return nd.Activation(input, ActivationType.Relu);
                case "sigmoid":
                    return nd.Activation(input, ActivationType.Sigmoid);
                case "softrelu":
                    return nd.Activation(input, ActivationType.Softrelu);
                case "softsign":
                    return nd.Activation(input, ActivationType.Softsign);
                case "leakyrely":
                    return nd.LeakyReLU(input);
            }

            return input;
        }
    }
}