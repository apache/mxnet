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
    public class RNNCell : BaseRNNCell
    {
        public ActivationType _activation;

        private Symbol _hB;

        private Symbol _hW;

        private Symbol _iB;

        private Symbol _iW;

        public RNNCell(int num_hidden, ActivationType activation = ActivationType.Tanh, string prefix = "rnn_", RNNParams @params = null) : base(prefix, @params)
        {
            this._num_hidden = num_hidden;
            this._activation = activation;
            this._iW = this.Params.Get("i2h_weight");
            this._iB = this.Params.Get("i2h_bias");
            this._hW = this.Params.Get("h2h_weight");
            this._hB = this.Params.Get("h2h_bias");
        }

        public override StateInfo[] StateInfo
        {
            get
            {
                return new StateInfo[] 
                {
                    new StateInfo(){ Shape = new Shape(0, _num_hidden), Layout = "NC" },
                };
            }
        }

        public override string[] GateNames
        {
            get
            {
                return new string[] { "" };
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            this._counter += 1;
            var name = $"{this._prefix}t{this._counter}_";
            var i2h = sym.FullyConnected(data: inputs, weight: this._iW, bias: this._iB, num_hidden: this._num_hidden, symbol_name: $"{name}i2h");
            var h2h = sym.FullyConnected(data: states[0], weight: this._hW, bias: this._hB, num_hidden: this._num_hidden, symbol_name: $"{name}h2h");
            var output = GetActivation(i2h + h2h, this._activation, name: $"{name}out");
            return (output, output);
        }
    }
}
