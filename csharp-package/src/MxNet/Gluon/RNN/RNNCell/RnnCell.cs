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
using MxNet.Initializers;
using MxNet.Sym.Numpy;

namespace MxNet.Gluon.RNN
{
    public class RNNCell : HybridRecurrentCell
    {
        private readonly string _activation;
        private readonly int _hidden_size;
        private int _input_size;

        public RNNCell(int hidden_size, string activation = "tanh", string i2h_weight_initializer = null,
            string h2h_weight_initializer = null,
            string i2h_bias_initializer = "zeros", string h2h_bias_initializer = "zeros", int input_size = 0) : base()
        {
            _hidden_size = hidden_size;
            _activation = activation;
            _input_size = input_size;
            this["i2h_weight"] = Params.Get("i2h_weight", shape: new Shape(hidden_size, input_size),
                init: Initializer.Get(i2h_weight_initializer), allow_deferred_init: true);
            this["h2h_weight"] = Params.Get("h2h_weight", shape: new Shape(hidden_size, hidden_size),
                init: Initializer.Get(h2h_weight_initializer), allow_deferred_init: true);
            this["i2h_bias"] = Params.Get("i2h_bias", shape: new Shape(hidden_size),
                init: Initializer.Get(i2h_bias_initializer), allow_deferred_init: true);
            this["h2h_bias"] = Params.Get("h2h_bias", shape: new Shape(hidden_size),
                init: Initializer.Get(h2h_bias_initializer), allow_deferred_init: true);
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return new[] {new StateInfo {Layout = "NC", Shape = new Shape(batch_size, _hidden_size)}};
        }

        public override string Alias()
        {
            return "rnn";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            var prefix = $"t{_counter}_";
            var states = args[0];
            var i2h_weight = args[1];
            var h2h_weight = args[2];
            var i2h_bias = args[3];
            var h2h_bias = args[4];
            NDArrayOrSymbol output = null;

            if (x.IsNDArray)
            {
                var i2h = nd.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size);
                var h2h = nd.FullyConnected(states, h2h_weight, h2h_bias, _hidden_size);
                var i2h_plus_h2h = nd.ElemwiseAdd(i2h, h2h);
                output = Activation(i2h_plus_h2h, _activation);
            }
            else
            {
                var i2h = sym.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size, symbol_name: prefix + "i2h");
                var h2h = sym.FullyConnected(states, h2h_weight, h2h_bias, _hidden_size, symbol_name: prefix + "h2h");
                var i2h_plus_h2h = sym.ElemwiseAdd(i2h, h2h, prefix + "plus0");
                output = Activation(i2h_plus_h2h, _activation, name: prefix + "out");
            }

            return (output, new[] {output});
        }

        internal static StateInfo[] CellsStateInfo(RecurrentCell[] cells, int batch_size)
        {
            var ret = new List<StateInfo>();
            foreach (var item in cells) ret.AddRange(item.StateInfo(batch_size));

            return ret.ToArray();
        }

        internal static NDArrayOrSymbol[] CellsBeginState(RecurrentCell[] cells, int batch_size, string state_func,
            FuncArgs args = null)
        {
            var ret = new List<NDArrayOrSymbol>();
            foreach (var item in cells) ret.AddRange(item.BeginState(batch_size, state_func, args));

            return ret.ToArray();
        }

        internal static NDArrayOrSymbol[] GetBeginState(RecurrentCell cell, NDArrayOrSymbol[] begin_state,
            NDArrayOrSymbol inputs, int batch_size)
        {
            if (begin_state != null)
            {
                if (inputs.IsNDArray)
                {
                    var ctx = inputs.NdX.ctx;
                    var args = new FuncArgs();
                    args.Add("ctx", ctx);
                    begin_state = cell.BeginState(batch_size, "nd.Zeros", args);
                }
                else
                {
                    begin_state = cell.BeginState(batch_size, "sym.Zeros");
                }
            }

            return begin_state;
        }

        internal static NDArrayOrSymbol[] GetBeginState(RecurrentCell cell, NDArrayOrSymbol[] begin_state,
            NDArrayOrSymbol[] inputs, int batch_size)
        {
            if (begin_state != null)
            {
                if (inputs[0].IsNDArray)
                {
                    var ctx = inputs[0].NdX.ctx;
                    var args = new FuncArgs();
                    args.Add("ctx", ctx);
                    begin_state = cell.BeginState(batch_size, "nd.Zeros", args);
                }
                else
                {
                    begin_state = cell.BeginState(batch_size, "sym.Zeros");
                }
            }

            return begin_state;
        }

        internal static (NDArrayOrSymbol[], int, int) FormatSequence(int? length, NDArrayOrSymbol inputs, string layout,
            bool merge, string in_layout = null)
        {
            var axis = layout.IndexOf('T');
            var batch_axis = layout.IndexOf('N');
            var batch_size = 0;
            var in_axis = !string.IsNullOrWhiteSpace(in_layout) ? in_layout.IndexOf('T') : axis;
            NDArrayOrSymbol[] data_inputs = null;
            if (inputs.IsSymbol)
            {
                if (!merge)
                {
                    if (inputs.SymX.ListOutputs().Count != 1)
                        throw new Exception("unroll doesn't allow grouped symbol as input. Please convert " +
                                            "to list with list(inputs) first or let unroll handle splitting.");
                    data_inputs = new NDArrayOrSymbol[] {sym.Split(inputs.SymX, length.Value, in_axis, true)};
                }
            }
            else if (inputs.IsNDArray)
            {
                batch_size = inputs.NdX.shape[batch_axis];
                if (!merge)
                {
                    if (length != inputs.NdX.shape[in_axis])
                        throw new Exception("Invalid length!");

                    data_inputs = nd.Split(inputs.NdX, inputs.NdX.shape[in_axis], in_axis, true).NDArrayOrSymbols;
                }
            }

            return (data_inputs, axis, batch_size);
        }

        internal static (NDArrayOrSymbol[], int, int) FormatSequence(int? length, NDArrayOrSymbol[] inputs,
            string layout, bool merge, string in_layout = null)
        {
            var axis = layout.IndexOf('T');
            NDArrayOrSymbol data_inputs = null;
            if (inputs[0].IsSymbol)
                data_inputs = sym.Stack(inputs.ToList().ToSymbols(), inputs.Length, axis);
            else if (inputs[0].IsNDArray) data_inputs = nd.Stack(inputs.ToList().ToNDArrays(), inputs.Length, axis);

            return FormatSequence(length, data_inputs, layout, merge, in_layout);
        }

        internal static NDArrayOrSymbol[] MaskSequenceVariableLength(NDArrayOrSymbol data, int length,
            NDArrayOrSymbol valid_length, int time_axis, bool merge)
        {
            NDArrayOrSymbol outputs = null;
            NDArrayOrSymbol[] ret = null;
            if (data.IsNDArray)
                outputs = nd.SequenceMask(data, valid_length, true, time_axis);
            else
                outputs = sym.SequenceMask(data, valid_length, true, time_axis);

            if (!merge)
            {
                if (data.IsSymbol)
                    ret = new NDArrayOrSymbol[] {sym.Split(data.SymX, length, time_axis, true)};
                else if (data.IsNDArray) ret = nd.Split(data, length, time_axis, true).NDArrayOrSymbols;
            }

            return ret;
        }

        internal static NDArrayOrSymbol[] MaskSequenceVariableLength(NDArrayOrSymbol[] data, int length,
            NDArrayOrSymbol valid_length, int time_axis, bool merge)
        {
            NDArrayOrSymbol outputs = null;
            if (data[0].IsNDArray)
                outputs = nd.Stack(data.ToList().ToNDArrays(), data.Length, time_axis);
            else
                outputs = sym.Stack(data.ToList().ToSymbols(), data.Length, time_axis);

            return MaskSequenceVariableLength(outputs, length, valid_length, time_axis, merge);
        }

        internal static NDArrayOrSymbol[] _reverse_sequences(NDArrayOrSymbol[] sequences, int unroll_step,
            NDArrayOrSymbol valid_length = null)
        {
            NDArrayOrSymbol[] reversed_sequences = null;
            if (valid_length == null)
            {
                reversed_sequences = sequences;
                reversed_sequences.Reverse();
            }
            else
            {
                NDArrayOrSymbol seqRev = null;
                if (sequences[0].IsNDArray)
                    seqRev = nd.SequenceReverse(nd.Stack(sequences.ToList().ToNDArrays(), sequences.Length),
                        valid_length, true);
                else
                    seqRev = sym.SequenceReverse(sym.Stack(sequences.ToList().ToSymbols(), sequences.Length),
                        valid_length, true);


                if (unroll_step > 1 || sequences[0].IsSymbol)
                {
                    if (sequences[0].IsNDArray)
                        reversed_sequences = nd.Split(seqRev, unroll_step, 0, true).NDArrayOrSymbols;
                    else
                        reversed_sequences = new NDArrayOrSymbol[] {sym.Split(seqRev, unroll_step, 0, true)};
                }
                else
                {
                    reversed_sequences = new[] {seqRev};
                }
            }

            return reversed_sequences;
        }

        internal static (NDArrayOrSymbol[], NDArrayOrSymbol[]) DynamicUnroll(RecurrentCell cell, NDArrayOrSymbol[] inputs,
            NDArrayOrSymbol[] begin_state = null, int drop_inputs = 0, int drop_outputs = 0, string layout = "NTC", _Symbol valid_length = null)
        {
            throw new NotImplementedException();
        }
    }
}