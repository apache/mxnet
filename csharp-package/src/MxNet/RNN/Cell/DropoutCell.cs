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
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class DropoutCell : BaseRNNCell
    {
        private readonly float dropout;

        public DropoutCell(float dropout, string prefix, RNNParams @params = null) : base(prefix, @params)
        {
            this.dropout = dropout;
        }

        public override StateInfo[] StateInfo => new StateInfo[0];

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            if (this.dropout > 0)
            {
                inputs = sym.Dropout(data: inputs, p: this.dropout);
            }

            return (inputs, states);
        }

        public override (Symbol, SymbolList) Unroll(int length, SymbolList inputs, SymbolList begin_state = null, string layout = null, bool? merge_outputs = null)
        {
            this.Reset();
            var (ret, _) = __internals__.NormalizeSequence(length, inputs, layout, merge_outputs.Value);
            return (ret, new SymbolList());
        }
    }
}
