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
    public class ConvRNNCell : BaseConvRNNCell
    {
        public ConvRNNCell(Shape input_shape, int num_hidden, (int, int)? h2h_kernel = null, (int, int)? h2h_dilate = null,
            (int, int)? i2h_kernel = null, (int, int)? i2h_stride = null, (int, int)? i2h_pad = null, 
            (int, int)? i2h_dilate = null, Initializer i2h_weight_initializer = null, Initializer h2h_weight_initializer = null,
            Initializer i2h_bias_initializer = null, Initializer h2h_bias_initializer = null, RNNActivation activation = null,
            string prefix = "ConvLSTM_", RNNParams @params = null, string conv_layout = "NCHW") 
            : base(input_shape, num_hidden, h2h_kernel.HasValue ? h2h_kernel.Value : (3, 3),
                  h2h_dilate.HasValue ? h2h_dilate.Value : (1,1), i2h_kernel.HasValue ? i2h_kernel.Value : (3,3),
                  i2h_stride.HasValue ? i2h_stride.Value : (1,1), i2h_pad.HasValue ? i2h_pad.Value : (1,1),
                  i2h_dilate.HasValue ? i2h_dilate.Value : (1,1), i2h_weight_initializer, h2h_weight_initializer, 
                  i2h_bias_initializer, h2h_bias_initializer, activation != null ? activation : new RNNActivation("leaky"), prefix, @params)
        {
        }

        public override string[] GateNames => new string[] { "" };

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            this._counter += 1;
            var name = $"{_prefix}t{_counter}_";
            var _tup_1 = this.ConvForward(inputs, states, name);
            var i2h = _tup_1.Item1;
            var h2h = _tup_1.Item2;
            var output = _activation.Invoke(i2h + h2h);
            return (output, output);
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
