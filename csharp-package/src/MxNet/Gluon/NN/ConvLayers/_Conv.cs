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
using System.Collections.Generic;
using System.Linq;
using MxNet.Initializers;
using MxNet.Sym.Numpy;

namespace MxNet.Gluon.NN
{
    public class _Conv : HybridBlock
    {
        private string _op_name;

        public _Conv(int channels, int[] kernel_size, int[] strides, int[] padding, int[] dilation,
            int groups, string layout, int in_channels = 0, ActivationType? activation = null, bool use_bias = true,
            Initializer weight_initializer = null, string bias_initializer = "zeros", int[] adj = null,
            string op_name = "Convolution") : base()
        {
            NumFilter = channels;
            InChannels = in_channels;
            Strides = strides;
            Padding = padding;
            Dialation = dilation;
            _op_name = op_name;
            KernalSize = kernel_size;
            NumGroup = groups;
            Layout = layout;
            Activation = activation.HasValue ? new Activation(activation.Value) : null;
            UseBias = use_bias;
            WeightInitializer = weight_initializer;
            BiasInitializer = bias_initializer;
            Adj = adj;

            var dshape = new int[kernel_size.Length + 2];
            dshape[layout.ToCharArray().ToList().IndexOf('N')] = 1;
            dshape[layout.ToCharArray().ToList().IndexOf('C')] = in_channels;
            var wshapes = _infer_weight_shape(op_name, new Shape(dshape));

            this["weight"] = Params.Get("weight", OpGradReq.Write, wshapes[1], init: weight_initializer,
                allow_deferred_init: true);

            if (UseBias)
                this["bias"] = Params.Get("bias", OpGradReq.Write, wshapes[2], init: Initializer.Get(bias_initializer),
                    allow_deferred_init: true);
            else
                this["bias"] = null;
        }

        public int NumFilter { get; set; }

        public int InChannels { get; set; }

        public int[] KernalSize { get; set; }

        public int[] Strides { get; set; }

        public int[] Padding { get; set; }

        public int[] Dialation { get; set; }

        public int NumGroup { get; set; }

        public string Layout { get; set; }

        public Activation Activation { get; set; }

        public bool UseBias { get; set; }

        public Initializer WeightInitializer { get; set; }

        public Initializer BiasInitializer { get; set; }

        public int[] Adj { get; set; }

        internal Shape[] _infer_weight_shape(string op_name, Shape data_shape)
        {
            var conv = sym.Convolution(_Symbol.Var("data", shape: data_shape), null, kernel: new Shape(KernalSize),
                     num_filter: NumFilter,
                     stride: new Shape(Strides), dilate: new Shape(Dialation), pad: new Shape(Padding), no_bias: !UseBias,
                     num_group: NumGroup, bias: null);

            return conv.InferShapePartial(new Dictionary<string, Shape>()).Item1;
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            var weight = args[0];
            var bias = args.Length > 1 ? args[1] : null;
            if (x.IsNDArray)
                x = nd.Convolution(x, weight, bias, new Shape(KernalSize), NumFilter, new Shape(Strides),
                    new Shape(Dialation), new Shape(Padding),
                    NumGroup, 1024, !UseBias);
            else
                x = sym.Convolution(x, weight, bias, new Shape(KernalSize), NumFilter, new Shape(Strides),
                    new Shape(Dialation), new Shape(Padding),
                    NumGroup, 1024, !UseBias);

            if (Activation != null)
                return Activation.Call(x);

            return x;
        }

        public override string Alias()
        {
            return "conv";
        }

        public override string ToString()
        {
            var shape = Params["weight"].Shape;
            var mapping = $"{(shape.Dimension >= 2 && shape[1] > 0 ? shape[1].ToString() : "None")} -> {shape[0]}";
            var s = $"{this.GetType().Name}({mapping}, kernel_size=({string.Join(", ", KernalSize)}), stride=({string.Join(", ", Strides)})";
            var len_kernal_size = KernalSize.Length;
            if (!Padding.All(i => i == 0))
                s += $", padding=({string.Join(", ", Padding)})";
            if (!Dialation.All(i => i == 1))
                s += $", dilation=({string.Join(", ", Dialation)})";
            if (NumGroup != 1)
                s += $", groups={NumGroup}";
            if (!UseBias)
                s += ", bias=False";
            if (Activation != null)
                s += $", {Activation.Alias()}";
            s += ")";
            return s;
        }
    }
}