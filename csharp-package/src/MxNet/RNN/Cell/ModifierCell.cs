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
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class ModifierCell : BaseRNNCell
    {
        internal BaseRNNCell base_cell;

        public ModifierCell(BaseRNNCell base_cell) : base("", null)
        {
            base_cell._modified = true;
            this.base_cell = base_cell;
        }

        public override StateInfo[] StateInfo => this.base_cell.StateInfo;

        public override RNNParams Params
        {
            get
            {
                _own_params = false;
                return base_cell.Params;
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            throw new NotSupportedException();
        }

        public override SymbolList BeginState(string func = "zeros", FuncArgs kwargs = null)
        {
            Debug.Assert(!this._modified, "After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.");
            this.base_cell._modified = false;
            var begin = this.base_cell.BeginState(func, kwargs);
            this.base_cell._modified = true;
            return begin;
        }

        public override NDArrayDict UnpackWeights(NDArrayDict args)
        {
            return this.base_cell.UnpackWeights(args);
        }

        public override NDArrayDict PackWeights(NDArrayDict args)
        {
            return this.base_cell.PackWeights(args);
        }
    }
}
