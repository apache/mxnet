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
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class GRUCell : BaseRNNCell
    {
        private Symbol _hB;

        private Symbol _hW;

        private Symbol _iB;

        private Symbol _iW;

        public GRUCell(int num_hidden, string prefix = "lstm_", RNNParams @params = null) : base(prefix, @params)
        {
            this._num_hidden = num_hidden;
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
                return new string[] {
                    "_r",
                    "_z",
                    "_o"
                };
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            this._counter += 1;
            var seq_idx = this._counter;
            var name = $"{this._prefix}t{this._counter}_";
            var prev_state_h = states[0];
            var i2h = sym.FullyConnected(data: inputs, weight: this._iW, bias: this._iB, num_hidden: this._num_hidden * 3, symbol_name: String.Format("%s_i2h", name));
            var h2h = sym.FullyConnected(data: prev_state_h, weight: this._hW, bias: this._hB, num_hidden: this._num_hidden * 3, symbol_name: String.Format("%s_h2h", name));
            // pylint: disable=unbalanced-tuple-unpacking
            var _tup_1 = sym.SliceChannel(i2h, num_outputs: 3, symbol_name: $"{name}_i2h_slice");
            var i2h_r = _tup_1[0];
            var i2h_z = _tup_1[1];
            i2h = _tup_1[2];
            var _tup_2 = sym.SliceChannel(h2h, num_outputs: 3, symbol_name: $"{name}_h2h_slice");
            var h2h_r = _tup_2[0];
            var h2h_z = _tup_2[1];
            h2h = _tup_2[2];
            var reset_gate = sym.Activation(i2h_r + h2h_r, act_type: ActivationType.Sigmoid, symbol_name: $"{name}_r_act");
            var update_gate = sym.Activation(i2h_z + h2h_z, act_type: ActivationType.Sigmoid, symbol_name: $"{name}_z_act");
            var next_h_tmp = sym.Activation(i2h + reset_gate * h2h, act_type: ActivationType.Tanh, symbol_name: $"{name}_h_act");
            var next_h = sym.ElemwiseAdd((1 - update_gate) * next_h_tmp, update_gate * prev_state_h, symbol_name: $"{name}out");
            return (next_h, next_h);
        }
    }
}
