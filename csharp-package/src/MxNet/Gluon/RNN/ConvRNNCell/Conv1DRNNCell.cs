using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RNN
{
    public class Conv1DRNNCell : _ConvRNNCell
    {
        public Conv1DRNNCell(Shape input_shape, int hidden_channels, int i2h_kernel, int h2h_kernel, int i2h_pad = 0, 
            int i2h_dilate = 1, int h2h_dilate = 1, string i2h_weight_initializer = null, string h2h_weight_initializer = null,
            string i2h_bias_initializer = "zeros", string h2h_bias_initializer = "zeros", string conv_layout = "NCW", ActivationType activation = ActivationType.Tanh) 
            : base(input_shape, hidden_channels, new int[] { i2h_kernel } , new int[] { h2h_kernel } , new int[] { i2h_pad },
                  new int[] { i2h_dilate }, new int[] { h2h_dilate }, i2h_weight_initializer, h2h_weight_initializer, 
                  i2h_bias_initializer, h2h_bias_initializer, 1, conv_layout, activation)
        {
        }
    }
}
