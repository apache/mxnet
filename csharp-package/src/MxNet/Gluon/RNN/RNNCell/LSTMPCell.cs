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
using MxNet.Initializers;
using System;

namespace MxNet.Gluon.RNN
{
    public class LSTMPCell : HybridRecurrentCell
    {
        private readonly string _activation;
        private readonly int _hidden_size;
        private int _input_size;
        private readonly string _recurrent_activation;

        public LSTMPCell(int hidden_size, int projection_size,  string i2h_weight_initializer = null, string h2h_weight_initializer = null,
            string h2r_weight_initializer = null, string i2h_bias_initializer = "zeros", string h2h_bias_initializer = "zeros", int input_size = 0) : base()
        {
            _hidden_size = hidden_size;
            _input_size = input_size;
            this["i2h_weight"] = Params.Get("i2h_weight", shape: new Shape(hidden_size, input_size),
                init: Initializer.Get(i2h_weight_initializer), allow_deferred_init: true);
            this["h2h_weight"] = Params.Get("h2h_weight", shape: new Shape(hidden_size, hidden_size),
                init: Initializer.Get(h2h_weight_initializer), allow_deferred_init: true);
            this["i2h_bias"] = Params.Get("i2h_bias", shape: new Shape(hidden_size),
                init: Initializer.Get(i2h_bias_initializer), allow_deferred_init: true);
            this["h2h_bias"] = Params.Get("h2h_bias", shape: new Shape(hidden_size),
                init: Initializer.Get(h2h_bias_initializer), allow_deferred_init: true);

            throw new NotImplementedException();
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return new[]
            {
                new StateInfo {Layout = "NC", Shape = new Shape(batch_size, _hidden_size)},
                new StateInfo {Layout = "NC", Shape = new Shape(batch_size, _hidden_size)}
            };
        }

        public override string Alias()
        {
            return "lstmp";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            throw new NotImplementedException();
            /*
            var prefix = $"t{_counter}_";
            var states_0 = args[0];
            var states_1 = args[1];
            var i2h_weight = args[2];
            var h2h_weight = args[3];
            var i2h_bias = args[4];
            var h2h_bias = args[5];
            NDArrayOrSymbol next_c = null;
            NDArrayOrSymbol next_h = null;

            if (x.IsNDArray)
            {
                var i2h = nd.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size * 4);
                var h2h = nd.FullyConnected(states_0, h2h_weight, h2h_bias, _hidden_size * 4);
                var gates = nd.ElemwiseAdd(i2h, h2h);
                var slice_gates = nd.Split(gates, 4);
                var in_gate = Activation(slice_gates[0], _recurrent_activation);
                var forget_gate = Activation(slice_gates[1], _recurrent_activation);
                var in_transform = Activation(slice_gates[2], _activation);
                var out_gate = Activation(slice_gates[3], _recurrent_activation);
                next_c = nd.ElemwiseAdd(nd.ElemwiseMul(forget_gate, states_1),
                    nd.ElemwiseMul(in_gate, in_transform));
                next_h = nd.ElemwiseMul(out_gate, Activation(next_c.NdX, _activation));
            }
            else
            {
                var i2h = sym.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size * 4, symbol_name: prefix + "i2h");
                var h2h = sym.FullyConnected(states_0, h2h_weight, h2h_bias, _hidden_size * 4,
                    symbol_name: prefix + "h2h");
                var gates = sym.ElemwiseAdd(i2h, h2h, prefix + "plus0");
                var slice_gates = sym.Split(gates, 4, symbol_name: prefix + "slice");
                var in_gate = Activation(slice_gates[0], _recurrent_activation, name: prefix + "i");
                var forget_gate = Activation(slice_gates[1], _recurrent_activation, name: prefix + "f");
                var in_transform = Activation(slice_gates[2], _activation, name: prefix + "c");
                var out_gate = Activation(slice_gates[3], _recurrent_activation, name: prefix + "o");
                next_c = sym.ElemwiseAdd(sym.ElemwiseMul(forget_gate, states_1, prefix + "mul0"),
                    sym.ElemwiseMul(in_gate, in_transform, prefix + "mul1"), prefix + "state");
                next_h = sym.ElemwiseMul(out_gate, Activation(next_c.SymX, _activation, name: prefix + "i2h"),
                    prefix + "out");
            }

            return (next_h, new[] {next_h, next_c});
            */
        }
    }
}