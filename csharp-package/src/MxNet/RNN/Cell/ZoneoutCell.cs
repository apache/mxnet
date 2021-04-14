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
using System.Linq;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class ZoneoutCell : ModifierCell
    {
        public Symbol prev_output;

        public float zoneout_outputs;

        public float zoneout_states;

        public ZoneoutCell(BaseRNNCell base_cell, float zoneout_outputs= 0, float zoneout_states= 0) : base(base_cell)
        {
            Debug.Assert(!(base_cell is FusedRNNCell), "FusedRNNCell doesn't support zoneout. Please unfuse first.");
            Debug.Assert(!(base_cell is BidirectionalCell), "BidirectionalCell doesn't support zoneout since it doesn't support step. Please add ZoneoutCell to the cells underneath instead.");
            Debug.Assert(!(base_cell is SequentialRNNCell), "Bidirectional SequentialRNNCell doesn't support zoneout. Please add ZoneoutCell to the cells underneath instead.");
            this.zoneout_outputs = zoneout_outputs;
            this.zoneout_states = zoneout_states;
            this.prev_output = null;
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            var cell = this.base_cell;
            var p_outputs = this.zoneout_outputs;
            var p_states = this.zoneout_states;
            var _tup_1 = cell.Call(inputs, states);
            var next_output = _tup_1.Item1;
            var next_states = _tup_1.Item2;
            Func<float, Symbol, Symbol> mask = (p, like) => sym.Dropout(sym.OnesLike(like), p: p);
            var prev_output = this.prev_output != null ? this.prev_output : sym.Zeros(new Shape(0, 0));
            var output = p_outputs != 0.0 ? sym.Where(mask(p_outputs, next_output), next_output, prev_output) : next_output;

            if(p_states != 0)
            {
                states = Enumerable.Zip(next_states, states, (new_s, old_s) => 
                {
                    return sym.Where(mask(p_states, new_s), new_s, old_s);
                }).ToList();
            }
            else
            {
                states = next_states;
            }

            this.prev_output = output;
            return (output, states);
        }

        public override void Reset()
        {
            base.Reset();
            this.prev_output = null;
        }
    }
}
