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

namespace MxNet.Gluon.RNN
{
    public class BidirectionalCell : HybridRecurrentCell
    {
        private readonly string _output_prefix;

        public BidirectionalCell(RecurrentCell l_cell, RecurrentCell r_cell, string output_prefix = "bi_") : base()
        {
            RegisterChild(l_cell, "l_cell");
            RegisterChild(r_cell, "r_cell");
            _output_prefix = output_prefix;
        }

        public (_Symbol, List<SymbolList>) Call(_Symbol inputs, List<SymbolList> states)
        {
            throw new NotSupportedException("Bidirectional cannot be stepped. Please use unroll");
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return RNNCell.CellsStateInfo(_childrens.Values.ToArray(), batch_size);
        }

        public override NDArrayOrSymbolList BeginState(int batch_size = 0, string func = null, FuncArgs args = null)
        {
            return RNNCell.CellsBeginState(_childrens.Values.ToArray(), batch_size, func);
        }

        public override (NDArrayOrSymbolList, NDArrayOrSymbolList) Unroll(int length, NDArrayOrSymbolList inputs,
            NDArrayOrSymbolList begin_state = null, string layout = "NTC", bool? merge_outputs = null,
            _Symbol valid_length = null)
        {
            Reset();
            var axis = 0;
            var batch_size = 0;
            (inputs, axis, batch_size) = RNNCell.FormatSequence(length, inputs, layout, false);
            var reversed_inputs = RNNCell._reverse_sequences(inputs, length, valid_length);
            begin_state = RNNCell.GetBeginState(this, begin_state, inputs, batch_size);
            var states = begin_state.ToList();
            var l_cell = _childrens["l_cell"];
            var r_cell = _childrens["r_cell"];

            var (l_outputs, l_states) = l_cell.Unroll(length, inputs, states.Take(l_cell.StateInfo().Length).ToList(),
                layout, merge_outputs, valid_length);
            var (r_outputs, r_states) = r_cell.Unroll(length, inputs, states.Skip(l_cell.StateInfo().Length).ToList(),
                layout, merge_outputs, valid_length);

            var reversed_r_outputs = RNNCell._reverse_sequences(r_outputs, length, valid_length);
            if (!merge_outputs.HasValue)
            {
                merge_outputs = l_outputs.Length > 1;

                (l_outputs, _, _) = RNNCell.FormatSequence(null, l_outputs, layout, merge_outputs.Value);
                (reversed_r_outputs, _, _) =
                    RNNCell.FormatSequence(null, reversed_r_outputs, layout, merge_outputs.Value);
            }

            NDArrayOrSymbolList outputs = null;
            if (merge_outputs.Value)
            {
                if (reversed_r_outputs[0].IsNDArray)
                    reversed_r_outputs = new NDArrayOrSymbolList
                        {nd.Stack(reversed_r_outputs.ToNDArrays(), reversed_r_outputs.Length, axis)};
                else
                    reversed_r_outputs = new NDArrayOrSymbolList
                        {sym.Stack(reversed_r_outputs.ToSymbols(), reversed_r_outputs.Length, axis)};

                var concatList = l_outputs;
                concatList.Add(reversed_r_outputs);
                if (reversed_r_outputs[0].IsNDArray)
                    outputs = new NDArrayOrSymbolList {nd.Concat(concatList.ToNDArrays(), 2)};
                else
                    outputs = new NDArrayOrSymbolList {sym.Concat(concatList.ToSymbols(), 2)};
            }
            else
            {
                var outputs_temp = new NDArrayOrSymbolList();
                for (var i = 0; i < l_outputs.Length; i++)
                {
                    var l_o = l_outputs[i];
                    var r_o = reversed_r_outputs[i];
                    if (l_o.IsNDArray)
                        outputs_temp.Add(nd.Concat(new ndarray[] {l_o, r_o}));
                    else
                        outputs_temp.Add(sym.Concat(new SymbolList {l_o, r_o}, 1, symbol_name: $"{_output_prefix}t{i}"));
                }

                outputs = outputs_temp;
                outputs_temp = new NDArrayOrSymbolList();
            }


            if (valid_length != null)
                outputs = RNNCell.MaskSequenceVariableLength(outputs, length, valid_length, axis, merge_outputs.Value);

            states.Clear();
            states.AddRange(l_states);
            states.AddRange(r_states);

            return (outputs, states);
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            return default;
        }
    }
}