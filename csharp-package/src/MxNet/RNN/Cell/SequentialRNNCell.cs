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
    public class SequentialRNNCell : BaseRNNCell
    {
        private List<BaseRNNCell> _cells;

        public bool _override_cell_params;

        public SequentialRNNCell(RNNParams @params = null) : base("", @params)
        {
            this._override_cell_params = @params != null;
            this._cells = new List<BaseRNNCell>();
        }

        public override StateInfo[] StateInfo => __internals__.CellsStateInfo(this._cells.ToArray());

        public void Add(BaseRNNCell cell)
        {
            this._cells.Add(cell);
            if (this._override_cell_params)
            {
                Debug.Assert(cell._own_params, "Either specify params for SequentialRNNCell or child cells, not both.");
                foreach (var item in this.Params._params)
                {
                    cell.Params._params[item.Key] = item.Value;
                }
            }

            foreach (var item in cell.Params._params)
            {
                this.Params._params[item.Key] = item.Value;
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            this._counter += 1;
            var next_states = new SymbolList();
            var p = 0;
            foreach (var cell in this._cells)
            {
                Debug.Assert(!(cell is BidirectionalCell));
                var n = cell.StateInfo.Length;
                var state = states.Skip(p).Take(p + n).ToArray();
                p += n;
                var _tup_1 = cell.Call(inputs, state);
                inputs = _tup_1.Item1;
                state = _tup_1.Item2;
                next_states.Add(state);
            }

            next_states.Add(null);
            return (inputs, next_states);
        }

        public override SymbolList BeginState(string func = "zeros", FuncArgs kwargs = null)
        {
            Debug.Assert(!this._modified, "After applying modifier cells (e.g. ZoneoutCell) the base cell cannot be called directly. Call the modifier cell instead.");

            return __internals__.CellsBeginState(this._cells.ToArray(), func, kwargs);
        }

        public override NDArrayDict UnpackWeights(NDArrayDict args)
        {
            return __internals__.CellsUnpackWeights(this._cells.ToArray(), args);
        }

        public override NDArrayDict PackWeights(NDArrayDict args)
        {
            return __internals__.CellsPackWeights(this._cells.ToArray(), args);
        }

        public override (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = null, bool? merge_outputs = null)
        {
            this.Reset();
            var num_cells = this._cells.Count;
            if (begin_state == null)
            {
                begin_state = this.BeginState();
            }

            var p = 0;
            var next_states = new SymbolList();
            foreach (var _tup_1 in this._cells.Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1)))
            {
                var i = _tup_1.Item1;
                var cell = _tup_1.Item2;
                var n = cell.StateInfo.Length;
                var states = begin_state.Skip(p).Take(p + n).ToArray();
                p += n;
                var _tup_2 = cell.Unroll(length, inputs: inputs, begin_state: states, layout: layout, merge_outputs: i < num_cells - 1 ? null : merge_outputs);
                inputs = _tup_2.Item1;
                states = _tup_2.Item2;
                next_states.Add(states);
            }

            return (inputs, next_states);
        }
    }
}
