using MxNet.Initializers;
using MxNet.Sym.Numpy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MxNet.Gluon.NN
{
    public class DeformableConvolution : HybridBlock
    {
        public int _channels;

        public int _in_channels;

        public Activation act;

        public Parameter deformable_conv_bias;

        public Parameter deformable_conv_weight;

        public Parameter offset_bias;

        public Parameter offset_weight;

        public int offset_channels;

        public int[] kernel;

        public int[] strides;

        public int[] padding;

        public int[] dilation;

        public int num_filter;

        public int num_group;

        public bool no_bias;

        public string layout;

        public int num_deformable_group;

        public int[] adj;

        public DeformableConvolution(int channels, (int, int)? kernel_size = null, (int, int)? strides = null, (int, int)? padding = null, (int, int)? dilation = null,
            int groups = 1, int num_deformable_group = 1, string layout = "NCHW", bool use_bias = true, int in_channels = 0, ActivationType? activation = null,
            Initializer weight_initializer = null, string bias_initializer = "zeros", string offset_weight_initializer = "zeros",
            string offset_bias_initializer = "zeros", bool offset_use_bias = true, int[] adj = null, string op_name = "DeformableConvolution")
        {
            this._channels = channels;
            this._in_channels = in_channels;
            Debug.Assert(new string[] { "NCHW", "NHWC" }.Contains(layout), "Only supports 'NCHW' and 'NHWC' layout for now");
            var offset_channels = 2 * kernel_size.Value.Item1 * kernel_size.Value.Item2 * num_deformable_group;

            this.kernel = kernel_size.HasValue ? new int[] { kernel_size.Value.Item1, kernel_size.Value.Item2 } : new int[] { 1, 1 };
            this.strides = strides.HasValue ? new int[] { strides.Value.Item1, strides.Value.Item2 } : new int[] { 1, 1 };
            this.padding = padding.HasValue ? new int[] { padding.Value.Item1, padding.Value.Item2 } : new int[] { 0, 0 };
            this.dilation = dilation.HasValue ? new int[] { dilation.Value.Item1, dilation.Value.Item2 } : new int[] { 0, 0 };
            this.num_filter = offset_channels;
            this.num_group = groups;
            this.no_bias = !offset_use_bias;
            this.layout = layout;
            this.num_deformable_group = num_deformable_group;
            this.adj = adj;
            var dshape = new int[kernel.Length + 2];
            dshape[layout.IndexOf('N')] = 1;
            dshape[layout.IndexOf('C')] = in_channels;
            
            
            var offsetshapes = _infer_weight_shape("convolution", new Shape(dshape));
            this.offset_weight = new Parameter("offset_weight", shape: offsetshapes[1], init: Initializer.Get(offset_weight_initializer), allow_deferred_init: true);
            if (offset_use_bias)
            {
                this.offset_bias = new Parameter("offset_bias", shape: offsetshapes[2], init: Initializer.Get(offset_bias_initializer), allow_deferred_init: true);
            }
            else
            {
                this.offset_bias = null;
            }
            var deformable_conv_weight_shape = new int[kernel.Length + 2];
            deformable_conv_weight_shape[0] = channels;
            deformable_conv_weight_shape[2] = kernel[0];
            deformable_conv_weight_shape[3] = kernel[1];
            this.deformable_conv_weight = new Parameter("deformable_conv_weight", shape: new Shape(deformable_conv_weight_shape), init: weight_initializer, allow_deferred_init: true);
            if (use_bias)
            {
                this.deformable_conv_bias = new Parameter("deformable_conv_bias", shape: new Shape(channels), init: bias_initializer, allow_deferred_init: true);
            }
            else
            {
                this.deformable_conv_bias = null;
            }

            if (activation.HasValue)
            {
                this.act = new Activation(activation.Value);
            }
            else
            {
                this.act = null;
            }

            this["deformable_conv_bias"] = deformable_conv_bias;
            this["deformable_conv_weight"] = deformable_conv_weight;
            this["offset_bias"] = offset_bias;
            this["offset_weight"] = offset_weight;
        }

        public override string Alias()
        {
            return "deformable_conv";
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            //object act;
            //object offset;
            //if (offset_bias == null)
            //{
            //    offset = F.convolution(x, offset_weight, kernel: kernel, stride: strides, cudnn_off: true, this._kwargs_offset);
            //}
            //else
            //{
            //    offset = F.convolution(x, offset_weight, offset_bias, cudnn_off: true, this._kwargs_offset);
            //}
            //if (deformable_conv_bias == null)
            //{
            //    act = F.npx.deformable_convolution(data: x, offset: offset, weight: deformable_conv_weight, name: "fwd", this._kwargs_deformable_conv);
            //}
            //else
            //{
            //    act = F.npx.deformable_convolution(data: x, offset: offset, weight: deformable_conv_weight, bias: deformable_conv_bias, name: "fwd", this._kwargs_deformable_conv);
            //}

            //if (this.act)
            //{
            //    using (var np_array(true))
            //    {
            //        act = this.act(act);
            //    }
            //}

            //return is_np_array() ? act : act.as_nd_ndarray();

            throw new NotImplementedException();
        }

        internal Shape[] _infer_weight_shape(string op_name, Shape data_shape)
        {
            var conv = sym.Convolution(_Symbol.Var("data", shape: data_shape), null, kernel: new Shape(kernel),
                     num_filter: num_filter,
                     stride: new Shape(strides), dilate: new Shape(dilation), pad: new Shape(padding), no_bias: no_bias,
                     num_group: num_group, bias: null);

            return conv.InferShapePartial(new Dictionary<string, Shape>()).Item1;
        }
    }
}
