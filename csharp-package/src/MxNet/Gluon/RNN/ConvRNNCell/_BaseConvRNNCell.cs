using MxNet.Gluon.NN;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MxNet.Gluon.RNN
{
    public class _BaseConvRNNCell : HybridRecurrentCell
    {
        public int NumGates
        {
            get
            {

                return GateNames.Length;
            }
        }

        public virtual string[] GateNames { get; }

        public int _hidden_channels;
        public int _channel_axis;
        public int _in_channels;
        public Shape _input_shape;
        public Shape _state_shape;
        public string _conv_layout;
        public ActivationType _activation;
        public int[] _i2h_kernel;
        public int[] _stride;
        public int[] _i2h_pad;
        public int[] _i2h_dilate;
        public int[] _h2h_kernel;
        public int[] _h2h_dilate;
        public int[] _h2h_pad;

        public _BaseConvRNNCell(Shape input_shape, int hidden_channels, int[] i2h_kernel, int[] h2h_kernel,
                 int[] i2h_pad, int[] i2h_dilate, int[] h2h_dilate, string i2h_weight_initializer, string h2h_weight_initializer,
                 string i2h_bias_initializer, string h2h_bias_initializer, int dims, string conv_layout, ActivationType activation)
        {
            this._hidden_channels = hidden_channels;
            this._input_shape = input_shape;
            this._conv_layout = conv_layout;
            this._activation = activation;
            // Convolution setting
            Debug.Assert((from spec in new List<int[]>() {
                                i2h_kernel,
                                i2h_pad,
                                i2h_dilate,
                                h2h_kernel,
                                h2h_dilate
                            } select spec.Length == dims).All(x => x),
                    $"For {dims}D convolution, the convolution settings can only be either int or list/tuple of length {dims}");

            var strideTemp = new List<int>();
            for(int i = 0; i< dims; i++)
            {
                strideTemp.Add(1);
            }

            this._i2h_kernel = i2h_kernel;
            this._stride = strideTemp.ToArray();
            this._i2h_pad = i2h_pad;
            this._i2h_dilate = i2h_dilate;
            this._h2h_kernel = h2h_kernel;

            Debug.Assert((from k in this._h2h_kernel select k % 2 == 1).All(x => x), 
                            $"Only support odd number, get h2h_kernel= {new Shape(h2h_kernel)}");

            this._h2h_dilate = h2h_dilate;
            var _tup_1 = this.DecideShapes();
            this._channel_axis = _tup_1.Item1;
            this._in_channels = _tup_1.Item2;
            var i2h_param_shape = _tup_1.Item3;
            var h2h_param_shape = _tup_1.Item4;
            this._h2h_pad = _tup_1.Item5;
            this._state_shape = _tup_1.Item6;

            this["i2h_weight"] = new Parameter("i2h_weight", shape: i2h_param_shape, init: i2h_weight_initializer, allow_deferred_init: true);
            this["h2h_weight"] = new Parameter("h2h_weight", shape: h2h_param_shape, init: h2h_weight_initializer, allow_deferred_init: true);
            this["i2h_bias"] = new Parameter("i2h_bias", shape: new Shape(hidden_channels * this.NumGates), init: i2h_bias_initializer, allow_deferred_init: true);
            this["h2h_bias"] = new Parameter("h2h_bias", shape: new Shape(hidden_channels * this.NumGates), init: h2h_bias_initializer, allow_deferred_init: true);
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbolList) HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            throw new NotSupportedException();
        }

        public override StateInfo[] StateInfo(int batch_size = 0)
        {
            throw new NotSupportedException();
        }

        private (int, int, Shape, Shape, int[], Shape) DecideShapes()
        {
            List<int> dimensions = new List<int>();
            var channel_axis = this._conv_layout.IndexOf('C');
            var input_shape = this._input_shape;
            var in_channels = input_shape[channel_axis - 1];
            var hidden_channels = this._hidden_channels;
            if (channel_axis == 1)
            {
                dimensions = input_shape.Data.Skip(1).ToList();
            }
            else
            {
                dimensions = input_shape.Data.Take(input_shape.Dimension - 1).ToList();
            }

            var total_out = hidden_channels * this.NumGates;
            var i2h_param_shape = new Shape(total_out);
            var h2h_param_shape = new Shape(total_out);
            var state_shape = new Shape(hidden_channels);
            var conv_out_size = _get_conv_out_size(dimensions, this._i2h_kernel, this._i2h_pad, this._i2h_dilate);
            List<int> h2h_pad = new List<int>();
            for (int i = 0;i < this._h2h_dilate.Length; i++)
            {
                var d = this._h2h_dilate[i];
                var k = this._h2h_kernel[i];
                h2h_pad.Add(d * (k - 1) / 2);
            }

            if (channel_axis == 1)
            {
                i2h_param_shape.Add(in_channels);
                i2h_param_shape.Add(this._i2h_kernel);

                h2h_param_shape.Add(hidden_channels);
                h2h_param_shape.Add(this._h2h_kernel);

                state_shape.Add(conv_out_size);
            }
            else
            {
                i2h_param_shape.Add(this._i2h_kernel);
                i2h_param_shape.Add(in_channels);

                h2h_param_shape.Add(this._h2h_kernel);
                h2h_param_shape.Add(hidden_channels);
                state_shape.Insert(0, conv_out_size);
            }

            return (channel_axis, in_channels, i2h_param_shape, h2h_param_shape, h2h_pad.ToArray(), state_shape);
        }

        public static int[] _get_conv_out_size(List<int> dimensions, int[] kernels, int[] paddings, int[] dilations)
        {
            List<int> result = new List<int>();
            for(int i = 0; i< dimensions.Count; i++)
            {
                var x = dimensions[i];
                var k = kernels[i];
                var p = paddings[i];
                var d = dilations[i];

                if (x > 0)
                    result.Add(Convert.ToInt32(Math.Floor(x + 2d * p - d * (k - 1) - 1) + 1));
                else
                    result.Add(0);
            }

            return result.ToArray();
        }

        public NDArrayOrSymbolList ConvForward(NDArrayOrSymbol inputs, NDArrayOrSymbolList states, NDArrayOrSymbol i2h_weight,
                                                NDArrayOrSymbol h2h_weight, NDArrayOrSymbol i2h_bias, NDArrayOrSymbol h2h_bias, string prefix)
        {
            var i2h = F.convolution(data: inputs, num_filter: this._hidden_channels * this.NumGates, kernel: this._i2h_kernel, stride: this._stride, pad: this._i2h_pad, dilate: this._i2h_dilate, weight: i2h_weight, bias: i2h_bias, layout: this._conv_layout);
            var h2h = F.convolution(data: states[0], num_filter: this._hidden_channels * this.NumGates, kernel: this._h2h_kernel, dilate: this._h2h_dilate, pad: this._h2h_pad, stride: this._stride, weight: h2h_weight, bias: h2h_bias, layout: this._conv_layout);
            return new NDArrayOrSymbolList() { i2h, h2h };
        }
    }
}
