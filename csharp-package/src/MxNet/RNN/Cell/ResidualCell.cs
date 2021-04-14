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
using MxNet.RNN.Cell;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class ResidualCell : ModifierCell
    {
        public ResidualCell(BaseRNNCell base_cell) : base(base_cell)
        {
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            Symbol output = null;
            (output, states) = this.base_cell.Call(inputs, states);
            output = sym.ElemwiseAdd(output, inputs, symbol_name: $"{output.Name}_plus_residual");
            return (output, states);
        }

        public override (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = null, bool? merge_outputs = null)
        {
            this.Reset();
            this.base_cell._modified = false;
            var (outputs, states) = this.base_cell.Unroll(length, inputs: inputs, begin_state: begin_state, layout: layout, merge_outputs: merge_outputs);
            this.base_cell._modified = true;
            merge_outputs = merge_outputs == null ? outputs is Symbol : merge_outputs;
            var _tup_2 = __internals__.NormalizeSequence(length, inputs, layout, merge_outputs.Value);
            inputs = _tup_2.Item1;
            if (merge_outputs.Value)
            {
                outputs = sym.ElemwiseAdd(outputs, inputs, symbol_name: $"{outputs.Name}_plus_residual");
            }
            else
            {
                var outputsList = Enumerable.Zip(outputs.ToList(), inputs.ToList(), (output_sym, input_sym) =>
                {
                    return sym.ElemwiseAdd(output_sym, input_sym, symbol_name: $"{output_sym.Name}_plus_residual");
                }).ToList();

                outputs = Symbol.Group(outputs);
            }

            return (outputs, states);
        }
    }
}
