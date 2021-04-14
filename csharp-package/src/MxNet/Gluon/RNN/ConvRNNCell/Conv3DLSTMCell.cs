using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RNN
{
    public class Conv3DLSTMCell : _ConvRNNCell
    {
        public Conv3DLSTMCell(Shape input_shape, int hidden_channels, (int, int, int) i2h_kernel, (int, int, int) h2h_kernel, (int, int, int)? i2h_pad = null,
            (int, int, int)? i2h_dilate = null, (int, int, int)? h2h_dilate = null, string i2h_weight_initializer = null, string h2h_weight_initializer = null,
            string i2h_bias_initializer = "zeros", string h2h_bias_initializer = "zeros", string conv_layout = "NCDHW", ActivationType activation = ActivationType.Tanh) 
            : base(input_shape, hidden_channels, new int[] { i2h_kernel.Item1, i2h_kernel.Item2, i2h_kernel.Item3 } , new int[] { h2h_kernel.Item1, h2h_kernel.Item2, h2h_kernel.Item3 },
                  i2h_pad.HasValue ? new int[] { i2h_pad.Value.Item1, i2h_pad.Value.Item2, i2h_pad.Value.Item3 } : new int[] { 0, 0, 0 },
                  i2h_dilate.HasValue ? new int[] { i2h_dilate.Value.Item1, i2h_dilate.Value.Item2, i2h_dilate.Value.Item3 } : new int[] { 1, 1, 1 },
                  h2h_dilate.HasValue ? new int[] { h2h_dilate.Value.Item1, h2h_dilate.Value.Item2, h2h_dilate.Value.Item3 } : new int[] { 1, 1, 1 },
                  i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, 3, conv_layout, activation)
        {
        }
    }
}
