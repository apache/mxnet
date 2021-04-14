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
using MxNet.RNN.Cell;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class BidirectionalCell : BaseRNNCell
    {
        public string _output_prefix;

        public bool _override_cell_params;

        public BidirectionalCell(BaseRNNCell l_cell, BaseRNNCell r_cell, string output_prefix = "bi_", 
            RNNParams @params = null) : base("", @params)
        {
            this._output_prefix = output_prefix;
            this._override_cell_params = @params != null;
            if (this._override_cell_params)
            {
                Debug.Assert(l_cell._own_params != null && r_cell._own_params, "Either specify params for BidirectionalCell or child cells, not both.");
                foreach (var item in this.Params._params)
                {
                    l_cell.Params._params[item.Key] = item.Value;
                    r_cell.Params._params[item.Key] = item.Value;
                }
            }

            foreach (var item in l_cell.Params._params)
            {
                this.Params._params[item.Key] = item.Value;
            }

            foreach (var item in r_cell.Params._params)
            {
                this.Params._params[item.Key] = item.Value;
            }

            this._cells.Add(l_cell);
            this._cells.Add(r_cell);
        }

        public override StateInfo[] StateInfo => __internals__.CellsStateInfo(this._cells.ToArray());

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            throw new NotSupportedException("Bidirectional cannot be stepped. Please use unroll");
        }

        public override NDArrayDict UnpackWeights(NDArrayDict args)
        {
            return __internals__.CellsUnpackWeights(this._cells.ToArray(), args);
        }

        public override NDArrayDict PackWeights(NDArrayDict args)
        {
            return __internals__.CellsPackWeights(this._cells.ToArray(), args);
        }

        public override SymbolList BeginState(string func = "zeros", FuncArgs kwargs = null)
        {
            Debug.Assert(!this._modified, "After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.");
            return __internals__.CellsBeginState(this._cells.ToArray(), func, kwargs);
        }

        public override (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = null, bool? merge_outputs = null)
        {
            this.Reset();
            var _tup_1 = __internals__.NormalizeSequence(length, inputs, layout, false);
            inputs = _tup_1.Item1;
            var axis = _tup_1.Item2;
            if (begin_state == null)
            {
                begin_state = this.BeginState();
            }

            var states = begin_state;
            var l_cell = _cells[0];
            var r_cell = _cells[1];
            SymbolList l_outputs = null;
            SymbolList l_states = null;
            SymbolList r_outputs = null;
            SymbolList r_states = null;

            (l_outputs, l_states) = l_cell.Unroll(length, inputs: inputs, begin_state: states.Take(l_cell.StateInfo.Length).ToArray(), layout: layout, merge_outputs: merge_outputs);
            (r_outputs, r_states) = r_cell.Unroll(length, inputs: inputs.Reverse().ToList(), begin_state: states[l_cell.StateInfo.Length], layout: layout, merge_outputs: merge_outputs);
            if (merge_outputs == null)
            {
                merge_outputs = l_outputs.Length == 1 && r_outputs.Length == 1;
                if (!merge_outputs.Value)
                {
                    l_outputs = sym.SliceChannel(l_outputs, axis: axis, num_outputs: length, squeeze_axis: true).ToList();
                    r_outputs = sym.SliceChannel(r_outputs, axis: axis, num_outputs: length, squeeze_axis: true).ToList();
                }
            }

            if (merge_outputs.Value)
            {
                r_outputs = sym.Reverse(r_outputs, axis: axis);
            }
            else
            {
                r_outputs = r_outputs.Reverse().ToList();
            }

            SymbolList outputs = new SymbolList();

            for(int i = 0; i<l_outputs.Length; i++)
            {
                var l_o = l_outputs[i];
                var r_o = r_outputs[i];
                var lr = sym.Concat(new SymbolList(l_o, r_o), dim: 1 + (merge_outputs.Value ? 1 : 0), symbol_name: merge_outputs.Value ? $"{_output_prefix}out" : $"{_output_prefix}t{i}");
                outputs.Add(lr);
            }

            states = new List<Symbol> {
                l_states,
                r_states
            };

            if (merge_outputs.Value)
            {
                outputs = outputs[0];
            }
            
            return (outputs, states);
        }
    }
}
