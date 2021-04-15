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
using System;
using MxNet.Gluon.NN;

namespace MxNet.Gluon.ModelZoo.Vision
{
    public class RELU6 : HybridBlock
    {
        public RELU6(string prefix = "", ParameterDict @params = null) : base()
        {
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            var x = args[0];
            if (x.IsNDArray)
                return nd.Clip(x, 0, 6);

            return sym.Clip(x, 0, 6);
        }
    }

    public class LinearBottleneck : HybridBlock
    {
        private readonly HybridSequential output;
        private readonly bool use_shortcut;

        public LinearBottleneck(int in_channels, int channels, int t, int stride, string prefix = null,
            ParameterDict @params = null) : base()
        {
            use_shortcut = stride == 1 && in_channels == channels;
            output = new HybridSequential();
            MobileNet.AddConv(output, in_channels * t, relu6: true);
            MobileNet.AddConv(output, in_channels * t, 3, stride, 1, in_channels * t, relu6: true);
            MobileNet.AddConv(output, channels, active: false, relu6: true);

            RegisterChild(output, "out");
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var @out = output.Call(x, args);
            if (use_shortcut)
            {
                if (x.IsNDArray)
                    @out = nd.ElemwiseAdd(@out, x.NdX);
                else
                    @out = sym.ElemwiseAdd(@out, x.SymX);
            }

            return @out;
        }
    }

    public class MobileNet : HybridBlock
    {
        private static readonly int[] channels = {64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024};
        private static readonly int[] dw_channels = {32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024};
        private static readonly int[] strides = {1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1};

        public MobileNet(float multiplier = 1, int classes = 1000, string prefix = "", ParameterDict @params = null) :
            base()
        {
            Features = new HybridSequential();
            AddConv(Features, Convert.ToInt32(32 * multiplier), 3, pad: 1, stride: 2);
            for (var i = 0; i < dw_channels.Length; i++)
            {
                var dwc = Convert.ToInt32(multiplier * dw_channels[i]);
                var c = Convert.ToInt32(multiplier * channels[i]);
                var s = strides[i];

                AddConvDW(Features, dwc, c, s);
            }

            Features.Add(new GlobalAvgPool2D());
            Features.Add(new Flatten());
            RegisterChild(Features, "features");

            Output = new Dense(classes);
            RegisterChild(Output, "output");
        }

        public HybridSequential Features { get; set; }
        public Dense Output { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            x = Features.Call(x, args);
            x = Output.Call(x, args);
            return x;
        }

        internal static void AddConv(HybridSequential @out, int channels = 1, int kernel = 1, int stride = 1,
            int pad = 0,
            int num_group = 1, bool active = true, bool relu6 = false)
        {
            @out.Add(new Conv2D(channels, (kernel, kernel), (stride, stride), (pad, pad), groups: num_group,
                use_bias: false));
            @out.Add(new _BatchNorm());
            if (active)
            {
                if (relu6)
                    @out.Add(new RELU6());
                else
                    @out.Add(new Activation(ActivationType.Relu));
            }
        }

        internal static void AddConvDW(HybridSequential @out, int dw_channels, int channels = 1, int stride = 1,
            bool relu6 = false)
        {
            AddConv(@out, dw_channels, 3, stride, 1, dw_channels, relu6: relu6);
            AddConv(@out, channels, relu6: relu6);
        }

        public static MobileNet GetMobileNet(float multiplier, bool pretrained = false, Context ctx = null,
            string root = "", int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            var net = new MobileNet(multiplier, classes);

            var version_suffix = string.Format("{0:0.00}", multiplier);
            if (version_suffix == "1.00" || version_suffix == "0.50")
                version_suffix = version_suffix.Substring(0, 3);

            if (pretrained) net.LoadParameters(ModelStore.GetModelFile($"mobilenet{version_suffix}"), ctx);

            return net;
        }

        public static MobileNet MobileNet1_0(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNet(1, pretrained, ctx, root);
        }

        public static MobileNet MobileNet0_75(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNet(0.75f, pretrained, ctx, root);
        }

        public static MobileNet MobileNet0_5(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNet(0.5f, pretrained, ctx, root);
        }

        public static MobileNet MobileNet0_25(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNet(0.25f, pretrained, ctx, root);
        }
    }

    public class MobileNetV2 : HybridBlock
    {
        private static readonly int[] channels_group =
            {16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320};

        private static readonly int[] in_channels_group =
            {32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160};

        private static readonly int[] strides = {1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1};
        private static readonly int[] ts = {1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

        public MobileNetV2(float multiplier = 1, int classes = 1000, string prefix = "", ParameterDict @params = null)
            : base()
        {
            Features = new HybridSequential();
            MobileNet.AddConv(Features, Convert.ToInt32(32 * multiplier), 3, pad: 1, stride: 2, relu6: true);
            for (var i = 0; i < in_channels_group.Length; i++)
            {
                var in_c = Convert.ToInt32(multiplier * in_channels_group[i]);
                var c = Convert.ToInt32(multiplier * channels_group[i]);
                var s = strides[i];
                var t = ts[i];

                Features.Add(new LinearBottleneck(in_c, c, t, s));
            }

            var last_channel = multiplier > 1 ? Convert.ToInt32(1280 * multiplier) : 1280;
            MobileNet.AddConv(Features, last_channel, relu6: true);

            Features.Add(new GlobalAvgPool2D());

            RegisterChild(Features, "features");

            Output = new HybridSequential();
            Output.Add(new Conv2D(classes, (1, 1), use_bias: false));
            Output.Add(new Flatten());

            RegisterChild(Output, "output");
        }

        public HybridSequential Features { get; set; }
        public HybridSequential Output { get; set; }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList inputs)
        {
            inputs = Features.Call(inputs);
            inputs = Output.Call(inputs);
            return inputs;
        }

        public static MobileNetV2 GetMobileNetV2(float multiplier, bool pretrained = false, Context ctx = null,
            string root = "", int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            var net = new MobileNetV2(multiplier, classes);

            var version_suffix = string.Format("{0:0.00}", multiplier);
            if (version_suffix == "1.00" || version_suffix == "0.50")
                version_suffix = version_suffix.Substring(0, 3);

            if (pretrained) net.LoadParameters(ModelStore.GetModelFile($"mobilenetv2_{version_suffix}"), ctx);

            return net;
        }

        public static MobileNetV2 MobileNetV2_1_0(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNetV2(1, pretrained, ctx, root);
        }

        public static MobileNetV2 MobileNetV2_0_75(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNetV2(0.75f, pretrained, ctx, root);
        }

        public static MobileNetV2 MobileNetV2_0_5(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNetV2(0.5f, pretrained, ctx, root);
        }

        public static MobileNetV2 MobileNetV2_0_25(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetMobileNetV2(0.25f, pretrained, ctx, root);
        }
    }
}