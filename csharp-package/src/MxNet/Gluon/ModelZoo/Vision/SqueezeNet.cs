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
    public class SqueezeNet : HybridBlock
    {
        public SqueezeNet(string version, int classes = 1000, string prefix = "", ParameterDict @params = null) :
            base()
        {
            if (version != "1.0" && version != "1.1")
                throw new NotSupportedException("Unsupported version");

            Features = new HybridSequential();
            if (version == "1.0")
            {
                Features.Add(new Conv2D(96, (7, 7), (2, 2)));
                Features.Add(new Activation(ActivationType.Relu));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(16, 64, 64));
                Features.Add(MakeFire(16, 64, 64));
                Features.Add(MakeFire(32, 128, 128));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(32, 128, 128));
                Features.Add(MakeFire(48, 192, 192));
                Features.Add(MakeFire(48, 192, 192));
                Features.Add(MakeFire(64, 256, 256));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(64, 256, 256));
            }
            else if (version == "1.1")
            {
                Features.Add(new Conv2D(64, (3, 3), (2, 2)));
                Features.Add(new Activation(ActivationType.Relu));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(16, 64, 64));
                Features.Add(MakeFire(16, 64, 64));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(32, 128, 128));
                Features.Add(MakeFire(32, 128, 128));
                Features.Add(new MaxPool2D((3, 3), (2, 2), ceil_mode: true));
                Features.Add(MakeFire(48, 192, 192));
                Features.Add(MakeFire(48, 192, 192));
                Features.Add(MakeFire(64, 256, 256));
                Features.Add(MakeFire(64, 256, 256));
            }

            Features.Add(new Dropout(0.5f));

            RegisterChild(Features, "features");

            Output = new HybridSequential();
            Output.Add(new Conv2D(classes, (1, 1)));
            Output.Add(new Activation(ActivationType.Relu));
            Output.Add(new AvgPool2D((13, 13)));
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

        private HybridSequential MakeFire(int squeeze_channels, int expand1x1_channels, int expand3x3_channels)
        {
            var output = new HybridSequential();
            output.Add(MakeFireConv(squeeze_channels, (1, 1)));

            var paths = new HybridConcatenate(1);
            paths.Add(MakeFireConv(expand1x1_channels, (1, 1)));
            paths.Add(MakeFireConv(expand3x3_channels, (3, 3), (1, 1)));

            output.Add(paths);

            return output;
        }

        private HybridSequential MakeFireConv(int channels, (int, int) kernel_size, (int, int)? padding = null)
        {
            var output = new HybridSequential();
            output.Add(new Conv2D(channels, kernel_size, padding: padding));
            output.Add(new Activation(ActivationType.Relu));

            return output;
        }

        public static SqueezeNet GetSqueezeNet(string version, bool pretrained = false, Context ctx = null,
            string root = "", int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            var net = new SqueezeNet(version, classes);
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile("squeezenet" + version), ctx);

            return net;
        }

        public static SqueezeNet SqueezeNet1_0(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            return GetSqueezeNet("1.0", pretrained, ctx, root, classes);
        }

        public static SqueezeNet SqueezeNet1_1(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            return GetSqueezeNet("1.1", pretrained, ctx, root, classes);
        }
    }
}