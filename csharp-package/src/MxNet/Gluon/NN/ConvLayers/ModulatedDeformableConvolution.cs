using MxNet.Initializers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.NN
{
    public class ModulatedDeformableConvolution : HybridBlock
    {
        public ModulatedDeformableConvolution(int channels, (int, int)? kernel_size = null, (int, int)? strides = null, (int, int)? padding = null, (int, int)? dilation = null,
            int groups = 1, int num_deformable_group = 1, string layout = "NCHW", bool use_bias = true, int in_channels = 0, ActivationType? activation = null,
            Initializer weight_initializer = null, string bias_initializer = "zeros", string offset_weight_initializer = "zeros", string offset_bias_initializer = "zeros", bool offset_use_bias = true, int[] adj = null,
            string op_name = "ModulatedDeformableConvolution")
        {
            throw new NotImplementedException();
        }

        public override string Alias()
        {
            return "modulated_deformable_conv";
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            return base.HybridForward(args);
        }
    }
}
