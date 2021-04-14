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
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class LSTMCell : BaseRNNCell
    {
        private Symbol _hB;

        private Symbol _hW;

        private Symbol _iB;

        private Symbol _iW;

        public LSTMCell(int num_hidden, string prefix = "lstm_", RNNParams @params = null, float forget_bias = 1) : base(prefix, @params)
        {
            this._num_hidden = num_hidden;
            this._iW = this.Params.Get("i2h_weight");
            this._iB = this.Params.Get("i2h_bias", init: new LSTMBias(forget_bias));
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
                    new StateInfo(){ Shape = new Shape(0, _num_hidden), Layout = "NC" },
                };
            }
        }

        public override string[] GateNames
        {
            get
            {
                return new string[] { "_i", "_f", "_c", "_o" };
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            this._counter += 1;
            var name = $"{this._prefix}t{this._counter}_";
            var i2h = sym.FullyConnected(data: inputs, weight: this._iW, bias: this._iB, num_hidden: this._num_hidden * 4, symbol_name: $"{name}i2h");
            var h2h = sym.FullyConnected(data: states[0], weight: this._hW, bias: this._hB, num_hidden: this._num_hidden * 4, symbol_name: $"{name}h2h");
            var gates = i2h + h2h;
            var slice_gates = sym.SliceChannel(gates, 4, symbol_name: $"{name}slice");
            var in_gate = sym.Activation(slice_gates[0], act_type: ActivationType.Sigmoid, symbol_name: $"{name}i");
            var forget_gate = sym.Activation(slice_gates[1], act_type: ActivationType.Sigmoid, symbol_name: $"{name}f");
            var in_transform = sym.Activation(slice_gates[2], act_type: ActivationType.Tanh, symbol_name: $"{name}c");
            var out_gate = sym.Activation(slice_gates[3], act_type: ActivationType.Sigmoid, symbol_name: $"{name}o");
            var next_c = sym.BroadcastAdd(forget_gate * states[1], in_gate * in_transform, symbol_name: $"{name}state");
            var next_h = sym.BroadcastMul(out_gate, sym.Activation(next_c, act_type: ActivationType.Tanh), symbol_name: $"{name}out");
            return (next_h, new SymbolList(next_h, next_c));
        }
    }
}
