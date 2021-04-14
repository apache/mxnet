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
using System.Diagnostics;
using System.Text;

namespace MxNet.RecurrentLayer
{
    public class BaseConvRNNCell : BaseRNNCell
    {
        internal RNNActivation _activation;
        internal ConvolutionLayout _conv_layout;
        internal (int, int) _h2h_dilate;
        internal (int, int) _h2h_kernel;
        internal (int, int) _h2h_pad;
        internal Symbol _hB;
        internal Symbol _hW;
        internal (int, int) _i2h_dilate;
        internal (int, int) _i2h_kernel;
        internal (int, int) _i2h_pad;
        internal (int, int) _i2h_stride;
        internal Symbol _iB;
        internal Shape _input_shape;
        internal Symbol _iW;
        internal Shape _state_shape;

        public BaseConvRNNCell(Shape input_shape, int num_hidden, (int, int) h2h_kernel, (int, int) h2h_dilate,
                                (int, int) i2h_kernel, (int, int) i2h_stride, (int, int) i2h_pad, (int, int) i2h_dilate,
                                Initializer i2h_weight_initializer, Initializer h2h_weight_initializer, Initializer i2h_bias_initializer,
                                Initializer h2h_bias_initializer, RNNActivation activation, string prefix = "", RNNParams @params = null, ConvolutionLayout conv_layout = ConvolutionLayout.NCHW) : base(prefix, @params)
        {
            this._h2h_kernel = h2h_kernel;
            Debug.Assert(this._h2h_kernel.Item1 % 2 == 1 && this._h2h_kernel.Item2 % 2 == 1, $"Only support odd number, get h2h_kernel= {h2h_kernel}");
            this._h2h_pad = (h2h_dilate.Item1 * (h2h_kernel.Item1 - 1) / 2, h2h_dilate.Item2 * (h2h_kernel.Item2 - 1) / 2);
            this._h2h_dilate = h2h_dilate;
            this._i2h_kernel = i2h_kernel;
            this._i2h_stride = i2h_stride;
            this._i2h_pad = i2h_pad;
            this._i2h_dilate = i2h_dilate;
            this._num_hidden = num_hidden;
            this._input_shape = input_shape;
            this._conv_layout = conv_layout;
            this._activation = activation;
            // Infer state shape
            var data = Symbol.Variable("data");
            var stateConv = sym.Convolution(data: data, weight: null, bias: null, num_filter: this._num_hidden, kernel: _i2h_kernel, stride: _i2h_stride, pad: this._i2h_pad, dilate: this._i2h_dilate, layout: conv_layout);
            this._state_shape = stateConv.InferShape(new Dictionary<string, Shape>() { {"data", input_shape } }).Item2[0];
            this._state_shape = new Shape(0, this._state_shape[1]);
            // Get params
            this._iW = this.Params.Get("i2h_weight", init: i2h_weight_initializer);
            this._hW = this.Params.Get("h2h_weight", init: h2h_weight_initializer);
            this._iB = this.Params.Get("i2h_bias", init: i2h_bias_initializer);
            this._hB = this.Params.Get("h2h_bias", init: h2h_bias_initializer);
        }

        public override StateInfo[] StateInfo
        {
            get
            {
                return new StateInfo[] 
                {
                    new StateInfo(){ Shape = this._state_shape, Layout = MxUtil.EnumToString<ConvolutionLayout>(_conv_layout, sym.ConvolutionLayoutConvert)},
                     new StateInfo(){ Shape = this._state_shape, Layout = MxUtil.EnumToString<ConvolutionLayout>(_conv_layout, sym.ConvolutionLayoutConvert)}
                };
            }
        }

        public int NumGates
        {
            get
            {
                return this.GateNames.Length;
            }
        }

        public override (Symbol, SymbolList) Call(Symbol inputs, SymbolList states)
        {
            throw new NotSupportedException("BaseConvRNNCell is abstract class for convolutional RNN");
        }

        internal (Symbol, Symbol) ConvForward(Symbol inputs, SymbolList states, string name)
        {
            var i2h = sym.Convolution(symbol_name: $"{name}i2h", data: inputs, num_filter: this._num_hidden * this.NumGates, kernel: this._i2h_kernel, stride: this._i2h_stride, pad: this._i2h_pad, dilate: this._i2h_dilate, weight: this._iW, bias: this._iB, layout: this._conv_layout);
            var h2h = sym.Convolution(symbol_name: $"{name}h2h", data: states[0], num_filter: this._num_hidden * this.NumGates, kernel: this._h2h_kernel, dilate: this._h2h_dilate, pad: this._h2h_pad, stride: (1, 1), weight: this._hW, bias: this._hB, layout: this._conv_layout);
            return (i2h, h2h);
        }
    }
}
