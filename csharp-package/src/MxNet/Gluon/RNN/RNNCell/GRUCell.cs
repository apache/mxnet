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

namespace MxNet.Gluon.RNN
{
    public class GRUCell : HybridRecurrentCell
    {
        private readonly int _hidden_size;
        private int _input_size;


        public GRUCell(int hidden_size, string i2h_weight_initializer = null, string h2h_weight_initializer = null,
            string i2h_bias_initializer = "zeros", string h2h_bias_initializer = "zeros", int input_size = 0) : base()
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
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            return new[] {new StateInfo {Layout = "NC", Shape = new Shape(batch_size, _hidden_size)}};
        }

        public override string Alias()
        {
            return "gru";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x,
            NDArrayOrSymbolList args)
        {
            var prefix = $"t{_counter}_";
            var prev_state_h = args[0];
            var i2h_weight = args[1];
            var h2h_weight = args[2];
            var i2h_bias = args[3];
            var h2h_bias = args[4];
            NDArrayOrSymbol next_h = null;

            if (x.IsNDArray)
            {
                var i2h = nd.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size * 3);
                var h2h = nd.FullyConnected(prev_state_h, h2h_weight, h2h_bias, _hidden_size * 3);
                var i2hsplit = nd.Split(i2h, 3);
                var i2h_r = i2hsplit[0];
                var i2h_z = i2hsplit[1];
                i2h = i2hsplit[2];

                var h2hsplit = nd.Split(h2h, 3);
                var h2h_r = h2hsplit[0];
                var h2h_z = h2hsplit[1];
                h2h = h2hsplit[2];

                var reset_gate = Activation(nd.ElemwiseAdd(i2h_r, h2h_r), "sigmoid");
                var update_gate = Activation(nd.ElemwiseAdd(i2h_z, h2h_z), "sigmoid");
                var next_h_tmp = Activation(nd.ElemwiseAdd(i2h,
                        nd.ElemwiseMul(reset_gate, h2h)),
                    "tanh");
                var ones = nd.OnesLike(update_gate);
                next_h = nd.ElemwiseAdd(nd.ElemwiseMul(nd.ElemwiseSub(ones, update_gate),
                        next_h_tmp),
                    nd.ElemwiseMul(update_gate, prev_state_h));
            }
            else
            {
                var i2h = sym.FullyConnected(x, i2h_weight, i2h_bias, _hidden_size * 3, symbol_name: prefix + "i2h");
                var h2h = sym.FullyConnected(prev_state_h, h2h_weight, h2h_bias, _hidden_size * 3,
                    symbol_name: prefix + "h2h");
                var i2hsplit = sym.Split(i2h, 3, symbol_name: prefix + "i2h_slice");
                var i2h_r = i2hsplit[0];
                var i2h_z = i2hsplit[1];
                i2h = i2hsplit[2];

                var h2hsplit = sym.Split(h2h, 3, symbol_name: prefix + "h2h_slice");
                var h2h_r = h2hsplit[0];
                var h2h_z = h2hsplit[1];
                h2h = h2hsplit[2];

                var reset_gate = Activation(sym.ElemwiseAdd(i2h_r, h2h_r, prefix + "plus0"), "sigmoid",
                    name: prefix + "r_act");
                var update_gate = Activation(sym.ElemwiseAdd(i2h_z, h2h_z, prefix + "plus1"), "sigmoid",
                    name: prefix + "z_act");
                var next_h_tmp = Activation(
                    sym.ElemwiseAdd(i2h, sym.ElemwiseMul(reset_gate, h2h, prefix + "mul0"), prefix + "plus2"),
                    "tanh", name: prefix + "h_act");
                var ones = sym.OnesLike(update_gate, prefix + "ones_like0");
                next_h = sym.ElemwiseAdd(sym.ElemwiseMul(sym.ElemwiseSub(ones, update_gate, prefix + "minus0"),
                        next_h_tmp, prefix + "mul1"),
                    sym.ElemwiseMul(update_gate, prev_state_h, prefix + "mul2"), prefix + "out");
            }

            return (next_h, new[] {next_h});
        }
    }
}