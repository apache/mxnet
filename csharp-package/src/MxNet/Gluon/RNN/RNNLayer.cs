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
using System.Linq;
using System.Text;

namespace MxNet.Gluon.RecurrentNN
{
    public abstract class RNNLayer : HybridBlock
    {
        public int _dir;

        public float _dropout;

        public DType _dtype;

        public int _gates;

        public Initializer _h2h_bias_initializer;

        public Initializer _h2h_weight_initializer;

        public Initializer _h2r_weight_initializer;

        public int _hidden_size;

        public Initializer _i2h_bias_initializer;

        public Initializer _i2h_weight_initializer;

        public int _input_size;

        public string _layout;

        public float? _lstm_state_clip_max;

        public float? _lstm_state_clip_min;

        public bool? _lstm_state_clip_nan;

        public string _mode;

        public int _num_layers;

        public int? _projection_size;

        public bool _use_sequence_length;

        public bool skip_states;

        public RNNLayer(int hidden_size, int num_layers, string layout, float dropout, bool bidirectional, int input_size,
                 Initializer i2h_weight_initializer, Initializer h2h_weight_initializer, Initializer i2h_bias_initializer, 
                 Initializer h2h_bias_initializer, string mode, int? projection_size, Initializer h2r_weight_initializer,
                 float? lstm_state_clip_min, float? lstm_state_clip_max, bool? lstm_state_clip_nan,
                 DType dtype, bool use_sequence_length= false) : base()
        {
            Debug.Assert(new string[] { "TNC", "NTC" }.Contains(layout), $"Invalid layout {layout}; must be one of ['TNC' or 'NTC']");
            this._hidden_size = hidden_size;
            this._projection_size = projection_size;
            this._num_layers = num_layers;
            this._mode = mode;
            this._layout = layout;
            this._dropout = dropout;
            this._dir = bidirectional ? 2 : 1;
            this._input_size = input_size;
            this._i2h_weight_initializer = i2h_weight_initializer;
            this._h2h_weight_initializer = h2h_weight_initializer;
            this._i2h_bias_initializer = i2h_bias_initializer;
            this._h2h_bias_initializer = h2h_bias_initializer;
            this._h2r_weight_initializer = h2r_weight_initializer;
            this._lstm_state_clip_min = lstm_state_clip_min;
            this._lstm_state_clip_max = lstm_state_clip_max;
            this._lstm_state_clip_nan = lstm_state_clip_nan;
            this._dtype = dtype;
            this._use_sequence_length = use_sequence_length;
            this.skip_states = false;
            this._gates = new Dictionary<string, int> {
                    {
                        "rnn_relu",
                        1},
                    {
                        "rnn_tanh",
                        1},
                    {
                        "lstm",
                        4},
                    {
                        "gru",
                        3
                }
            }[mode];

            var ng = this._gates;
            var ni = input_size;
            var nh = hidden_size;
            if (projection_size == null)
            {
                foreach (var i in Enumerable.Range(0, num_layers))
                {
                    foreach (var j in new List<string> { "l", "r" }.Take(_dir))
                    {
                        this.RegisterParam("{j}{i}_i2h_weight", shape: new Shape(ng * nh, ni), init: i2h_weight_initializer, dtype: dtype);
                        this.RegisterParam("{j}{i}_h2h_weight", shape: new Shape(ng * nh, nh), init: h2h_weight_initializer, dtype: dtype);
                        this.RegisterParam("{j}{i}_i2h_bias", shape: new Shape(ng * nh), init: i2h_bias_initializer, dtype: dtype);
                        this.RegisterParam("{j}{i}_h2h_bias", shape: new Shape(ng * nh), init: h2h_bias_initializer, dtype: dtype);
                    }
                    ni = nh * this._dir;
                }
            }
            else
            {
                var np = this._projection_size.Value;
                foreach (var i in Enumerable.Range(0, num_layers))
                {
                    foreach (var j in new List<string> { "l", "r" }.Take(_dir))
                    {
                        this.RegisterParam("{}{}_i2h_weight", shape: new Shape(ng * nh, ni), init: i2h_weight_initializer, dtype: dtype);
                        this.RegisterParam("{}{}_h2h_weight", shape: new Shape(ng * nh, np), init: h2h_weight_initializer, dtype: dtype);
                        this.RegisterParam("{}{}_i2h_bias", shape: new Shape(ng * nh), init: i2h_bias_initializer, dtype: dtype);
                        this.RegisterParam("{}{}_h2h_bias", shape: new Shape(ng * nh), init: h2h_bias_initializer, dtype: dtype);
                        this.RegisterParam("{}{}_h2r_weight", shape: new Shape(np, nh), init: h2r_weight_initializer, dtype: dtype);
                    }
                    ni = np * this._dir;
                }
            }
        }

        private Parameter RegisterParam(string name, Shape shape, Initializer init, DType dtype)
        {
            var p = new Parameter(name, shape: shape, init: init, allow_deferred_init: true, dtype: dtype);
            this[name] = p;
            return p;
        }

        public abstract StateInfo[] StateInfo(int batch_size = 0);

        public override void Cast(DType dtype)
        {
            base.Cast(dtype);
            this._dtype = dtype;
        }

        public virtual NDArrayOrSymbolList BeginState(int batch_size = 0, string func = null, FuncArgs args = null, bool is_symbol = false)
        {
            var states = new NDArrayOrSymbolList();
            foreach (var _tup_1 in this.StateInfo(batch_size).Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1)))
            {
                var i = _tup_1.Item1;
                var info = _tup_1.Item2;
                if (info != null)
                {
                    info.Update(args);
                }
                else
                {
                    info = new StateInfo(args);
                }

                NDArrayOrSymbol state = null;
                if(func == "zeros")
                {
                    state = F.zeros(info.Shape, info.DataType, ctx: info.Ctx, is_symbol: is_symbol);
                }
                else if (func == "ones")
                {
                    state = F.ones(info.Shape, info.DataType, ctx: info.Ctx, is_symbol: is_symbol);
                }

                states.Add(state);
            }

            return states;
        }

        public override NDArrayOrSymbolList Call(NDArrayOrSymbolList args)
        {
            var inputs = args[0];
            var states = args[1].List;
            var sequence_length = args[2];

            this.skip_states = states == null;
            if (states == null)
            {
                if (inputs.IsNDArray)
                {
                    var batch_size = inputs.NdX.shape[this._layout.IndexOf("N")];
                    states = this.BeginState(batch_size, "zeros", new FuncArgs() { { "ctx:", inputs.NdX.ctx }, { "dtype", inputs.NdX.dtype } });
                }
                else
                {
                    states = this.BeginState(0, "zeros", is_symbol: true);
                }
            }

            if (this._use_sequence_length)
            {
                return base.Call((inputs, states, sequence_length));
            }
            else
            {
                return base.Call((inputs, states, null));
            }
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            var inputs = args[0];
            var states = args[1].List;
            var sequence_length = args[2];
            if (inputs.IsNDArray)
            {
                var batch_size = inputs.NdX.shape[this._layout.IndexOf("N")];
                var stateInfos = this.StateInfo(batch_size);
                for(int i = 0; i< states.Length; i++)
                {
                    var state = states[i];
                    var info = stateInfos[i];
                    if (state.NdX.shape.ToString() != info.Shape.ToString())
                    {
                        throw new Exception($"Invalid recurrent state shape. Expecting {info.Shape}, got {state.NdX.shape}.");
                    }
                }
            }
            
            var @out = ForwardKernel(inputs, states, sequence_length);
            // out is (output, state)
            return this.skip_states ? @out.Item1 : @out;
        }

        private (NDArrayOrSymbol, NDArrayOrSymbolList) ForwardKernel(NDArrayOrSymbol inputs, NDArrayOrSymbolList states, NDArrayOrSymbol sequence_length)
        {
            NDArrayOrSymbol outputs;
            object rnn_args;
            object @params;
            
            if (this._layout == "NTC")
            {
                inputs = F.swapaxes(inputs, 0, 1);
            }

            if(_projection_size  == null)
            {

            }
            else
            {

            }

            throw new NotImplementedException();
        }
    }
}
