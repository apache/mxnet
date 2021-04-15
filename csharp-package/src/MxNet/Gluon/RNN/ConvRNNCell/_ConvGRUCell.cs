using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RNN.ConvRNNCell
{
    public class _ConvGRUCell : _BaseConvRNNCell
    {
        public override string[] GateNames
        {
            get
            {
                return new string[] { "_r", "_z", "_o" };
            }
        }


        public _ConvGRUCell(Shape input_shape, int hidden_channels, int[] i2h_kernel, int[] h2h_kernel, int[] i2h_pad, int[] i2h_dilate,
                            int[] h2h_dilate, string i2h_weight_initializer, string h2h_weight_initializer, string i2h_bias_initializer,
                            string h2h_bias_initializer, int dims, string conv_layout, ActivationType activation)
            : base(input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate, i2h_weight_initializer,
                  h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, dims, conv_layout, activation)
        {
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            var shape = new List<int>();
            shape.Add(batch_size);
            shape.AddRange(this._state_shape.Data);

            var ret = new List<StateInfo>();
            ret.Add(new StateInfo() { Shape = new Shape(shape), Layout = this._conv_layout });

            return ret.ToArray();
        }

        public override string Alias()
        {
            return "conv_gru";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var states = args[0];
            var i2h_weight = args[1];
            var h2h_weight = args[2];
            var i2h_bias = args[3];
            var h2h_bias = args[4];

            var prefix = $"t{this._counter}_";
            var _tup_1 = this.ConvForward(x, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, prefix);
            var i2h = _tup_1[0];
            var h2h = _tup_1[1];
            NDArrayOrSymbolList _tup_2 = null;
            if(x.IsNDArray)
                _tup_2 = nd.SliceChannel(i2h, num_outputs: 3, axis: this._channel_axis);
            else
                _tup_2 = sym.SliceChannel(i2h, num_outputs: 3, axis: this._channel_axis, symbol_name: prefix + "i2h_slice");
            
            var i2h_r = _tup_2[0];
            var i2h_z = _tup_2[1];
            i2h = _tup_2[2];

            NDArrayOrSymbolList _tup_3 = null;
            if (x.IsNDArray)
                _tup_2 = nd.SliceChannel(h2h, num_outputs: 3, axis: this._channel_axis);
            else
                _tup_2 = sym.SliceChannel(h2h, num_outputs: 3, symbol_name: prefix + "h2h_slice", axis: this._channel_axis);
            
            var h2h_r = _tup_3[0];
            var h2h_z = _tup_3[1];
            h2h = _tup_3[2];
            var reset_gate = F.activation(i2h_r + h2h_r, act_type: "sigmoid");
            var update_gate = F.activation(i2h_z + h2h_z, act_type: "sigmoid");
            var next_h_tmp = F.activation(i2h + reset_gate * h2h, this._activation);
            var next_h = F.add((1 - update_gate) * next_h_tmp, update_gate * states[0]);
            return (next_h, new NDArrayOrSymbolList {
                    next_h
                });

        }
    }
}
