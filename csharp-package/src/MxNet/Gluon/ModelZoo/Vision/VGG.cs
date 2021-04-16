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
using MxNet.Gluon.NN;
using MxNet.Initializers;

namespace MxNet.Gluon.ModelZoo.Vision
{
    public class VGG : HybridBlock
    {
        private static readonly Dictionary<int, (int[], int[])> vgg_spec = new Dictionary<int, (int[], int[])>
        {
            {11, (new[] {1, 1, 2, 2, 2}, new[] {64, 128, 256, 512, 512})},
            {13, (new[] {2, 2, 2, 2, 2}, new[] {64, 128, 256, 512, 512})},
            {16, (new[] {2, 2, 3, 3, 3}, new[] {64, 128, 256, 512, 512})},
            {19, (new[] {2, 2, 4, 4, 4}, new[] {64, 128, 256, 512, 512})}
        };

        public VGG(int[] layers, int[] filters, int classes = 1000, bool batch_norm = false, string prefix = null,
            ParameterDict @params = null) : base()
        {
            Features = MakeFeatures(layers, filters, batch_norm);
            Features.Add(new Dense(4096, ActivationType.Relu, weight_initializer: "normal"));
            Features.Add(new Dropout(0.5f));
            Features.Add(new Dense(4096, ActivationType.Relu, weight_initializer: "normal"));
            Features.Add(new Dropout(0.5f));

            RegisterChild(Features, "features");

            Output = new Dense(classes, weight_initializer: "normal");

            RegisterChild(Output, "output");
        }

        public HybridSequential Features { get; set; }
        public Dense Output { get; set; }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList inputs)
        {
            inputs = Features.Call(inputs);
            inputs = Output.Call(inputs);
            return inputs;
        }

        private HybridSequential MakeFeatures(int[] layers, int[] filters, bool batch_norm = false)
        {
            var featurizer = new HybridSequential();
            for (var i = 0; i < layers.Length; i++)
            {
                var num = layers[i];
                for (var j = 0; j < num; j++)
                {
                    featurizer.Add(new Conv2D(filters[i], (3, 3), padding: (1, 1),
                        weight_initializer: new Xavier("gaussian", "out", 2)));
                    if (batch_norm)
                        featurizer.Add(new _BatchNorm());

                    featurizer.Add(new Activation(ActivationType.Relu));
                }

                featurizer.Add(new MaxPool2D(strides: (2, 2)));
            }

            return featurizer;
        }

        public static VGG GetVgg(int num_layers, bool pretrained = false, Context ctx = null, string root = "",
            bool batch_norm = false)
        {
            var (layers, filters) = vgg_spec[num_layers];
            var net = new VGG(layers, filters, batch_norm: batch_norm);
            if (pretrained)
            {
                var batch_norm_suffix = batch_norm ? "_bn" : "";
                net.LoadParameters(ModelStore.GetModelFile($"vgg{num_layers}{batch_norm_suffix}", root), ctx);
            }

            return net;
        }

        public static VGG Vgg11(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(11, pretrained, ctx, root);
        }

        public static VGG Vgg13(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(13, pretrained, ctx, root);
        }

        public static VGG Vgg16(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(16, pretrained, ctx, root);
        }

        public static VGG Vgg19(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(19, pretrained, ctx, root);
        }

        public static VGG Vgg11_BN(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(11, pretrained, ctx, root, true);
        }

        public static VGG Vgg13_BN(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(13, pretrained, ctx, root, true);
        }

        public static VGG Vgg16_BN(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(16, pretrained, ctx, root, true);
        }

        public static VGG Vgg19_BN(bool pretrained = false, Context ctx = null, string root = "")
        {
            return GetVgg(19, pretrained, ctx, root, true);
        }
    }
}