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
using MxNet.Gluon.RNN;
using MxNet.Initializers;
using MxNet.RNN.Cell;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class FusedRNNCell : BaseRNNCell
    {
        public bool _bidirectional;

        public List<string> _directions;

        public float _dropout;

        public bool _get_next_state;

        public RNNMode _mode;

        public int _num_layers;

        public Symbol _parameter;

        public FusedRNNCell(int num_hidden, int num_layers= 1, RNNMode mode = RNNMode.Lstm, bool  bidirectional= false,
                 float dropout= 0, bool get_next_state= false, float forget_bias= 1, string prefix = null, RNNParams @params = null) : base(prefix == null ? mode + "_" : prefix, @params)
        {
            this._num_hidden = num_hidden;
            this._num_layers = num_layers;
            this._mode = mode;
            this._bidirectional = bidirectional;
            this._dropout = dropout;
            this._get_next_state = get_next_state;
            this._directions = bidirectional ? new List<string> {
                "l",
                "r"
            } : new List<string> {
                "l"
            };

            //var initializer = new FusedRNN(null, num_hidden, num_layers, mode, bidirectional, forget_bias);
            //this._parameter = this.Params.Get("parameters", init: initializer);
        }

        public override StateInfo[] StateInfo
        {
            get
            {
                var b = this._bidirectional ? 1 : 0 + 1;
                var n = (this._mode == RNNMode.Lstm) ? 1 : 0 + 1;
                List<StateInfo> result = new List<StateInfo>();
                foreach (var _ in Enumerable.Range(0, n))
                {
                    result.Add(new Gluon.RNN.StateInfo() { Shape = new Shape((b * this._num_layers), 0, this._num_hidden), Layout = "LNC" });
                }

                return result.ToArray();
            }
        }

        public override string[] GateNames
        {
            get
            {
                return new Dictionary<RNNMode, string[]> {
                    {
                        RNNMode.RnnRelu,
                        new string[] {
                            ""
                        }},
                    {
                        RNNMode.RnnTanh,
                        new string[] {
                            ""
                        }},
                    {
                        RNNMode.Lstm,
                        new string[] {
                            "_i",
                            "_f",
                            "_c",
                            "_o"
                        }},
                    {
                        RNNMode.Gru,
                        new string[] {
                            "_r",
                            "_z",
                            "_o"
                        }}}[this._mode];
            }
        }

        public int NumGates
        {
            get
            {
                return this.GateNames.Length;
            }
        }

        private NDArrayDict SliceWeight(NDArray arr, int li, int lh)
        {
            int size;
            string name;
            var args = new NDArrayDict();

            var gate_names = this.GateNames;
            var directions = this._directions;
            var b = directions.Count;
            var p = 0;
            foreach (var layer in Enumerable.Range(0, this._num_layers))
            {
                foreach (var direction in directions)
                {
                    foreach (var gate in gate_names)
                    {
                        name = $"{_prefix}{direction}{layer}_i2h{gate}_weight";
                        if (layer > 0)
                        {
                            size = b * lh * lh;
                            args[name] = arr[$"{p}:{(p + size)}"].Reshape(new Shape(lh, b * lh));
                        }
                        else
                        {
                            size = li * lh;
                            args[name] = arr[$"{p}:{(p + size)}"].Reshape(new Shape(lh, li));
                        }
                        p += size;
                    }
                    foreach (var gate in gate_names)
                    {
                        name = $"{_prefix}{direction}{layer}_h2h{gate}_weight";
                        size = (int)Math.Pow(lh, 2);
                        args[name] = arr[$"{p}:{(p + size)}"].Reshape(new Shape(lh, lh));
                        p += size;
                    }
                }
            }
            foreach (var layer in Enumerable.Range(0, this._num_layers))
            {
                foreach (var direction in directions)
                {
                    foreach (var gate in gate_names)
                    {
                        name = $"{_prefix}{direction}{layer}_i2h{gate}_bias";
                        args[name] = arr[$"{p}:{(p + lh)}"];
                        p += lh;
                    }
                    foreach (var gate in gate_names)
                    {
                        name = $"{_prefix}{direction}{layer}_h2h{gate}_bias";
                        args[name] = arr[$"{p}:{(p + lh)}"];
                        p += lh;
                    }
                }
            }

            Debug.Assert(p == arr.Size, "Invalid parameters size for FusedRNNCell");
            return args;
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            throw new NotSupportedException("FusedRNNCell cannot be stepped. Please use unroll");
        }

        public override NDArrayDict UnpackWeights(NDArrayDict args)
        {
            var arr = args[this._parameter.Name];
            var b = this._directions.Count;
            var m = this.NumGates;
            var h = this._num_hidden;
            var num_input = arr.Size / b / h / m - (this._num_layers - 1) * (h + b * h + 2) - h - 2;
            var nargs = this.SliceWeight(arr, num_input, this._num_hidden);
            var newargs = nargs.ToDictionary(_tup_1 => _tup_1.Key, _tup_1 => _tup_1.Value.Copy());
            foreach (var item in newargs)
            {
                args[item.Key] = item.Value;
            }

            return args;
        }

        public override NDArrayDict PackWeights(NDArrayDict args)
        {
            var b = this._bidirectional ? 1 : 0 + 1;
            var m = this.NumGates;
            var c = this.GateNames;
            var h = this._num_hidden;
            var w0 = args[$"{_prefix}l0_i2h{c[0]}_weight"];
            var num_input = w0.Shape[1];
            var total = (num_input + h + 2) * h * m * b + (this._num_layers - 1) * m * h * (h + b * h + 2) * b;
            var arr = nd.Zeros(new Shape(total), ctx: w0.Context, dtype: w0.DataType);
            var slices = this.SliceWeight(arr, num_input, h);

            foreach (var sw in this.SliceWeight(arr, num_input, h))
            {
                var x = sw.Value;
                x = args[sw.Key];
            }

            args[this._parameter.Name] = arr;
            return args;
        }

        public override (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = null, bool? merge_outputs = null)
        {
            Symbol outputs = null;
            this.Reset();
            int axis = 0;
            (inputs, axis) = __internals__.NormalizeSequence(length, inputs, layout, true);
            if (axis == 1)
            {
                Logger.Warning("NTC layout detected. Consider using TNC for FusedRNNCell for faster speed");
                inputs = sym.SwapAxis(inputs, dim1: 0, dim2: 1);
            }
            else
            {
                Debug.Assert(axis == 0, $"Unsupported layout {layout}");
            }
            if (begin_state == null)
            {
                begin_state = this.BeginState();
            }

            var states = begin_state;
            Symbol state = states[0];
            Symbol state_cell = null;
            if (this._mode == RNNMode.Lstm)
            {
                state_cell = states[1];
            }

            var rnn = sym.RNN(data: inputs, parameters: this._parameter, state, state_cell, state_size: this._num_hidden, num_layers: this._num_layers, bidirectional: this._bidirectional, p: this._dropout, state_outputs: this._get_next_state, mode: this._mode, symbol_name: this._prefix + "rnn");

            var attr = new Dictionary<string, string> {
                {
                    "__layout__",
                    "LNC"
                }
            };
            if (!this._get_next_state)
            {
                outputs = rnn;
                states = new SymbolList();
            }
            else if (this._mode == RNNMode.Lstm)
            {
                rnn[1].SetAttr(attr);
                rnn[2].SetAttr(attr);
                outputs = rnn[0];
                states = new SymbolList(rnn[1], rnn[2]);
            }
            else
            {
                rnn[1].SetAttr(attr);
                outputs = rnn[0];
                states = new SymbolList(rnn[1]);
            }
            if (axis == 1)
            {
                outputs = sym.SwapAxis(outputs, dim1: 0, dim2: 1);
            }
            var _tup_2 = __internals__.NormalizeSequence(length, outputs, layout, merge_outputs.Value);
            outputs = _tup_2.Item1;
            return (outputs, states);
        }

        public SequentialRNNCell Unfuse()
        {
            var stack = new SequentialRNNCell();
            var get_cell = new Dictionary<RNNMode, Func<string, BaseRNNCell>> {
                {
                    RNNMode.RnnRelu,
                    (cell_prefix) => new RNNCell(this._num_hidden, activation: ActivationType.Relu, prefix: cell_prefix)},
                {
                    RNNMode.RnnTanh,
                    (cell_prefix) => new RNNCell(this._num_hidden, activation: ActivationType.Tanh, prefix: cell_prefix)},
                {
                    RNNMode.Lstm,
                    (cell_prefix) => new LSTMCell(this._num_hidden, prefix: cell_prefix)},
                {
                    RNNMode.Gru,
                    (cell_prefix) => new GRUCell(this._num_hidden, prefix: cell_prefix)
                }
            }[this._mode];

            foreach (var i in Enumerable.Range(0, this._num_layers))
            {
                if (this._bidirectional)
                {
                    stack.Add(new BidirectionalCell(get_cell($"{_prefix}l{1}_"), get_cell($"{_prefix}r{i}_"), output_prefix: $"{_prefix}bi_l{i}_"));
                }
                else
                {
                    stack.Add(get_cell($"{_prefix}l{i}_"));
                }
                if (this._dropout > 0 && i != this._num_layers - 1)
                {
                    stack.Add(new DropoutCell(this._dropout, prefix: $"{_prefix}_dropout{i}_"));
                }
            }
            return stack;
        }
    }
}
