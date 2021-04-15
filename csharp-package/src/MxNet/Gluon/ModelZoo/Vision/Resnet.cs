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
using System.Collections.Generic;
using System.Linq;
using MxNet.Gluon.NN;

namespace MxNet.Gluon.ModelZoo.Vision
{
    public class ResNet
    {
        private static readonly Dictionary<int, (string, int[], int[])> resnet_spec =
            new Dictionary<int, (string, int[], int[])>
            {
                {18, ("basic_block", new[] {2, 2, 2, 2}, new[] {64, 64, 128, 256, 512})},
                {34, ("basic_block", new[] {3, 4, 6, 3}, new[] {64, 64, 128, 256, 512})},
                {50, ("bottle_neck", new[] {3, 4, 6, 3}, new[] {64, 256, 512, 1024, 2048})},
                {101, ("bottle_neck", new[] {3, 4, 23, 3}, new[] {64, 256, 512, 1024, 2048})},
                {152, ("bottle_neck", new[] {3, 8, 36, 3}, new[] {64, 256, 512, 1024, 2048})}
            };

        internal static Conv2D Conv3x3(int channels, int stride, int in_channels)
        {
            return new Conv2D(channels, (3, 3), (stride, stride), (1, 1), use_bias: false, in_channels: in_channels);
        }

        public static ResNetV1 GetResNetV1(int num_layers, bool pretrained = false, Context ctx = null,
            string root = "", int classes = 1000, bool thumbnail = false, string prefix = null,
            ParameterDict @params = null)
        {
            if (!resnet_spec.ContainsKey(num_layers))
                throw new Exception("Invalid number of layers");

            var (block_type, layers, channels) = resnet_spec[num_layers];

            var net = new ResNetV1(block_type, layers, channels);
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile($"resnet{num_layers}_v1", root), ctx);

            return net;
        }

        public static ResNetV2 GetResNetV2(int num_layers, bool pretrained = false, Context ctx = null,
            string root = "", int classes = 1000, bool thumbnail = false, string prefix = null,
            ParameterDict @params = null)
        {
            if (!resnet_spec.ContainsKey(num_layers))
                throw new Exception("Invalid number of layers");

            var (block_type, layers, channels) = resnet_spec[num_layers];

            var net = new ResNetV2(block_type, layers, channels);
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile($"resnet{num_layers}_v2", root), ctx);

            return net;
        }

        public static ResNetV1 ResNet18_v1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV1(18, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV1 ResNet34_v1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV1(34, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV1 ResNet50_v1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV1(50, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV1 ResNet101_v1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV1(101, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV1 ResNet152_v1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV1(152, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV2 ResNet18_v2(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV2(18, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV2 ResNet34_v2(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV2(34, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV2 ResNet50_v2(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV2(50, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV2 ResNet101_v2(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV2(101, pretrained, ctx, root, classes, thumbnail);
        }

        public static ResNetV2 ResNet152_v2(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000,
            bool thumbnail = false, string prefix = "", ParameterDict @params = null)
        {
            return GetResNetV2(152, pretrained, ctx, root, classes, thumbnail);
        }
    }

    public class BasicBlockV1 : HybridBlock
    {
        private readonly HybridSequential body;
        private readonly HybridSequential ds;

        public BasicBlockV1(int channels, int stride, bool downsample = false, int in_channels = 0,
            string prefix = "", ParameterDict @params = null) : base()
        {
            body = new HybridSequential();
            body.Add(ResNet.Conv3x3(channels, stride, in_channels));
            body.Add(new _BatchNorm());
            body.Add(new Activation(ActivationType.Relu));
            body.Add(ResNet.Conv3x3(channels, 1, channels));
            body.Add(new _BatchNorm());
            if (downsample)
            {
                ds = new HybridSequential();
                ds.Add(new Conv2D(channels, (1, 1), (stride, stride), use_bias: false, in_channels: in_channels));
                ds.Add(new _BatchNorm());
                RegisterChild(ds, "downsample");
            }
            else
            {
                ds = null;
            }

            RegisterChild(body, "body");
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList inputs)
        {
            var residual = inputs[0];

            inputs = body.Call(inputs);
            if (ds != null)
                residual = ds.Call(residual, inputs);

            if (x.IsNDArray)
                x = nd.Activation(x.NdX + residual.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX + residual.SymX, ActivationType.Relu);

            return x;
        }
    }

    public class BasicBlockV2 : HybridBlock
    {
        private readonly _BatchNorm bn1;
        private readonly _BatchNorm bn2;
        private readonly Conv2D conv1;
        private readonly Conv2D conv2;
        private readonly Conv2D ds;

        public BasicBlockV2(int channels, int stride, bool downsample = false, int in_channels = 0,
            string prefix = "", ParameterDict @params = null) : base()
        {
            bn1 = new _BatchNorm();
            conv1 = ResNet.Conv3x3(channels, stride, in_channels);
            bn2 = new _BatchNorm();
            conv2 = ResNet.Conv3x3(channels, 1, channels);
            RegisterChild(bn1, "bn1");
            RegisterChild(conv1, "conv1");
            RegisterChild(bn2, "bn2");
            RegisterChild(conv2, "conv2");

            if (downsample)
            {
                ds = new Conv2D(channels, (1, 1), (stride, stride), use_bias: false, in_channels: in_channels);
                RegisterChild(ds, "downsample");
            }
            else
                ds = null;
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var residual = x;
            x = bn1.Call(x, args);
            if (x.IsNDArray)
                x = nd.Activation(x.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX, ActivationType.Relu);

            if (ds != null)
                residual = ds.Call(x, args);

            x = conv1.Call(x, args);

            x = bn2.Call(x, args);
            if (x.IsNDArray)
                x = nd.Activation(x.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX, ActivationType.Relu);
            x = conv2.Call(x, args);

            if (x.IsNDArray)
                return x.NdX + residual.NdX;

            return x.SymX + residual.SymX;
        }
    }

    public class BottleneckV1 : HybridBlock
    {
        private readonly HybridSequential body;
        private readonly HybridSequential ds;

        public BottleneckV1(int channels, int stride, bool downsample = false, int in_channels = 0,
            string prefix = "", ParameterDict @params = null) : base()
        {
            var channel_one_fourth = Convert.ToInt32(channels / 4);
            body = new HybridSequential();
            body.Add(new Conv2D(channel_one_fourth, (1, 1), (stride, stride)));
            body.Add(new _BatchNorm());
            body.Add(new Activation(ActivationType.Relu));
            body.Add(ResNet.Conv3x3(channel_one_fourth, 1, channel_one_fourth));
            body.Add(new _BatchNorm());
            body.Add(new Activation(ActivationType.Relu));
            body.Add(new Conv2D(channels, (1, 1), (1, 1)));
            body.Add(new _BatchNorm());

            if (downsample)
            {
                ds = new HybridSequential();
                ds.Add(new Conv2D(channels, (1, 1), (stride, stride), use_bias: false, in_channels: in_channels));
                ds.Add(new _BatchNorm());
                RegisterChild(ds, "downsample");
            }
            else
            {
                ds = null;
            }

            RegisterChild(body, "body");
        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var residual = x;

            x = body.Call(x, args);
            if (ds != null)
                residual = ds.Call(residual, args);

            if (x.IsNDArray)
                x = nd.Activation(x.NdX + residual.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX + residual.SymX, ActivationType.Relu);

            return x;
        }
    }

    public class BottleneckV2 : HybridBlock
    {
        private readonly _BatchNorm bn1;
        private readonly _BatchNorm bn2;
        private readonly _BatchNorm bn3;
        private readonly Conv2D conv1;
        private readonly Conv2D conv2;
        private readonly Conv2D conv3;
        private readonly Conv2D ds;

        public BottleneckV2(int channels, int stride, bool downsample = false, int in_channels = 0,
            string prefix = "", ParameterDict @params = null) : base()
        {
            var channel_one_fourth = Convert.ToInt32(channels / 4);
            bn1 = new _BatchNorm();
            conv1 = new Conv2D(channel_one_fourth, (1, 1), (1, 1), use_bias: false);
            bn2 = new _BatchNorm();
            conv2 = ResNet.Conv3x3(channel_one_fourth, stride, channel_one_fourth);
            bn3 = new _BatchNorm();
            conv3 = new Conv2D(channels, (1, 1), (1, 1), use_bias: false);
            RegisterChild(bn1, "bn1");
            RegisterChild(conv1, "conv1");
            RegisterChild(bn2, "bn2");
            RegisterChild(conv2, "conv2");
            RegisterChild(bn3, "bn3");
            RegisterChild(conv3, "conv3");

            if (downsample)
            {
                ds = new Conv2D(channels, (1, 1), (stride, stride), use_bias: false, in_channels: in_channels);
                RegisterChild(ds, "downsample");
            }
            else
                ds = null;

        }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var residual = x;
            x = bn1.Call(x, args);
            if (x.IsNDArray)
                x = nd.Activation(x.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX, ActivationType.Relu);

            if (ds != null)
                residual = ds.Call(x, args);

            x = conv1.Call(x, args);

            x = bn2.Call(x, args);
            if (x.IsNDArray)
                x = nd.Activation(x.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX, ActivationType.Relu);
            x = conv2.Call(x, args);

            x = bn3.Call(x, args);
            if (x.IsNDArray)
                x = nd.Activation(x.NdX, ActivationType.Relu);
            else
                x = sym.Activation(x.SymX, ActivationType.Relu);
            x = conv3.Call(x, args);

            if (x.IsNDArray)
                return x.NdX + residual.NdX;

            return x.SymX + residual.SymX;
        }
    }

    public class ResNetV1 : HybridBlock
    {
        public ResNetV1(string block, int[] layers, int[] channels, int classes = 1000, bool thumbnail = false,
            string prefix = "", ParameterDict @params = null) : base()
        {
            if (layers.Length != channels.Length - 1)
                throw new Exception("layers.length should be equal to channels.length - 1");

            Features = new HybridSequential();
            if (thumbnail)
            {
                Features.Add(ResNet.Conv3x3(channels[0], 1, 0));
            }
            else
            {
                Features.Add(new Conv2D(channels[0], (7, 7), (2, 2), (3, 3), use_bias: false));
                Features.Add(new _BatchNorm());
                Features.Add(new Activation(ActivationType.Relu));
                Features.Add(new MaxPool2D((3, 3), (2, 2), (1, 1)));
            }

            for (var i = 0; i < layers.Length; i++)
            {
                var stride = i == 0 ? 1 : 2;
                var num_layer = layers[i];
                Features.Add(MakeLayer(block, num_layer, channels[i + 1], stride, i + 1, channels[i]));
            }

            Features.Add(new GlobalAvgPool2D());

            Output = new Dense(classes, in_units: channels.Last());

            RegisterChild(Features, "features");
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

        private HybridSequential MakeLayer(string block, int layers, int channels, int stride, int stage_index,
            int in_channels = 0)
        {
            var layer = new HybridSequential();
            if (block == "basic_block")
            {
                layer.Add(new BasicBlockV1(channels, stride, channels != in_channels, in_channels, ""));
                for (var i = 0; i < layers - 1; i++)
                    layer.Add(new BasicBlockV1(channels, 1, false, channels, ""));
            }
            else if (block == "bottle_neck")
            {
                layer.Add(new BottleneckV1(channels, stride, channels != in_channels, in_channels, ""));
                for (var i = 0; i < layers - 1; i++)
                    layer.Add(new BottleneckV1(channels, 1, false, channels, ""));
            }

            return layer;
        }
    }

    public class ResNetV2 : HybridBlock
    {
        public ResNetV2(string block, int[] layers, int[] channels, int classes = 1000, bool thumbnail = false,
            string prefix = "", ParameterDict @params = null) : base()
        {
            if (layers.Length != channels.Length - 1)
                throw new Exception("layers.length should be equal to channels.length - 1");

            Features = new HybridSequential();
            Features.Add(new _BatchNorm(scale: false, center: false));
            if (thumbnail)
            {
                Features.Add(ResNet.Conv3x3(channels[0], 1, 0));
            }
            else
            {
                Features.Add(new Conv2D(channels[0], (7, 7), (2, 2), (3, 3), use_bias: false));
                Features.Add(new _BatchNorm());
                Features.Add(new Activation(ActivationType.Relu));
                Features.Add(new MaxPool2D((3, 3), (2, 2), (1, 1)));
            }

            var in_channels = channels[0];
            for (var i = 0; i < layers.Length; i++)
            {
                var stride = i == 0 ? 1 : 2;
                var num_layer = layers[i];
                Features.Add(MakeLayer(block, num_layer, channels[i + 1], stride, i + 1, in_channels));
                in_channels = channels[i + 1];
            }

            Features.Add(new _BatchNorm());
            Features.Add(new Activation(ActivationType.Relu));
            Features.Add(new GlobalAvgPool2D());
            Features.Add(new Flatten());

            Output = new Dense(classes, in_units: in_channels);

            RegisterChild(Features, "features");
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

        private HybridSequential MakeLayer(string block, int layers, int channels, int stride, int stage_index,
            int in_channels = 0)
        {
            var layer = new HybridSequential();
            if (block == "basic_block")
            {
                layer.Add(new BasicBlockV2(channels, stride, channels != in_channels, in_channels, ""));
                for (var i = 0; i < layers - 1; i++)
                    layer.Add(new BasicBlockV2(channels, 1, false, channels, ""));
            }
            else if (block == "bottle_neck")
            {
                layer.Add(new BottleneckV2(channels, stride, channels != in_channels, in_channels, ""));
                for (var i = 0; i < layers - 1; i++)
                    layer.Add(new BottleneckV2(channels, 1, false, channels, ""));
            }

            return layer;
        }
    }
}