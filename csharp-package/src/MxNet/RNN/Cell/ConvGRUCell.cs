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
    public class ConvGRUCell : BaseConvRNNCell
    {
        public ConvGRUCell(Shape input_shape, int num_hidden, (int, int)? h2h_kernel = null, (int, int)? h2h_dilate = null,
           (int, int)? i2h_kernel = null, (int, int)? i2h_stride = null, (int, int)? i2h_pad = null,
           (int, int)? i2h_dilate = null, Initializer i2h_weight_initializer = null, Initializer h2h_weight_initializer = null,
           Initializer i2h_bias_initializer = null, Initializer h2h_bias_initializer = null, RNNActivation activation = null,
           string prefix = "ConvGRU_", RNNParams @params = null, string conv_layout = "NCHW")
           : base(input_shape, num_hidden, h2h_kernel.HasValue ? h2h_kernel.Value : (3, 3),
                 h2h_dilate.HasValue ? h2h_dilate.Value : (1, 1), i2h_kernel.HasValue ? i2h_kernel.Value : (3, 3),
                 i2h_stride.HasValue ? i2h_stride.Value : (1, 1), i2h_pad.HasValue ? i2h_pad.Value : (1, 1),
                 i2h_dilate.HasValue ? i2h_dilate.Value : (1, 1), i2h_weight_initializer, h2h_weight_initializer,
                 i2h_bias_initializer, h2h_bias_initializer, activation != null ? activation : new RNNActivation("leaky"), prefix, @params)
        {
            throw new NotImplementedException();
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
            var name = String.Format("%st%d_", this._prefix, seq_idx);
            var (i2h, h2h) = this.ConvForward(inputs, states, name);
            // pylint: disable=unbalanced-tuple-unpacking
            var _tup_2 = sym.SliceChannel(i2h, num_outputs: 3, symbol_name: $"{name}_i2h_slice");
            var i2h_r = _tup_2[0];
            var i2h_z = _tup_2[1];
            i2h = _tup_2[1];
            var _tup_3 = sym.SliceChannel(h2h, num_outputs: 3, symbol_name: $"{name}_h2h_slice");
            var h2h_r = _tup_3[0];
            var h2h_z = _tup_3[1];
            h2h = _tup_3[2];
            var reset_gate = sym.Activation(i2h_r + h2h_r, act_type: ActivationType.Sigmoid, symbol_name: $"{name}_r_act");
            var update_gate = sym.Activation(i2h_z + h2h_z, act_type: ActivationType.Sigmoid, symbol_name: $"{name}_z_act");
            var next_h_tmp = this._activation.Invoke(i2h + reset_gate * h2h, $"{name}_h_act");
            var next_h = sym.BroadcastAdd((1 - update_gate) * next_h_tmp, update_gate * states[0], symbol_name: $"{name}out");
            return (next_h, next_h);
        }

        public override StateInfo[] StateInfo
        {
            get
            {
                return new StateInfo[]
                {
                    new StateInfo(){ Shape = this._state_shape, Layout = MxUtil.EnumToString<ConvolutionLayout>(_conv_layout, sym.ConvolutionLayoutConvert)},
                };
            }
        }
    }
}
