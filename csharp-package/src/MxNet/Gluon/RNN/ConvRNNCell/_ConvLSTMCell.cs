using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RNN.ConvRNNCell
{
    public class _ConvLSTMCell : _BaseConvRNNCell
    {
        public override string[] GateNames
        {
            get
            {
                return new string[] { "_i", "_f", "_c", "_o" };
            }
        }

        public _ConvLSTMCell(Shape input_shape, int hidden_channels, int[] i2h_kernel, int[] h2h_kernel, int[] i2h_pad, int[] i2h_dilate,
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
            ret.Add(new StateInfo() { Shape = new Shape(shape), Layout = this._conv_layout });

            return ret.ToArray();
        }

        public override string Alias()
        {
            return "conv_lstm";
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
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
            var gates = i2h + h2h;
            NDArrayOrSymbolList slice_gates = null;
            if(x.IsNDArray)
                slice_gates = nd.SliceChannel(gates, num_outputs: 4, axis: this._channel_axis);
            else
                slice_gates = sym.SliceChannel(gates, num_outputs: 4, symbol_name: prefix + "slice", axis: this._channel_axis);

            var in_gate = F.activation(slice_gates[0], act_type: "sigmoid");
            var forget_gate = F.activation(slice_gates[1], act_type: "sigmoid");
            var in_transform = F.activation(slice_gates[2], this._activation);
            var out_gate = F.activation(slice_gates[3], act_type: "sigmoid");
            var next_c = F.add(forget_gate * states[1], in_gate * in_transform);
            var next_h = F.multiply(out_gate,F.activation(next_c, this._activation));
            return (next_h, new NDArrayOrSymbolList {
                    next_h,
                    next_c
                });
        }
    }
}
