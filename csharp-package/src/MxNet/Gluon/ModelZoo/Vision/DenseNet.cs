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
using MxNet.Gluon.NN;

namespace MxNet.Gluon.ModelZoo.Vision
{
    public class DenseNet : HybridBlock
    {
        private static readonly Dictionary<int, (int, int, int[])> densenet_spec =
            new Dictionary<int, (int, int, int[])>
            {
                {121, (64, 32, new[] {6, 12, 24, 16})},
                {161, (96, 48, new[] {6, 12, 36, 24})},
                {169, (64, 32, new[] {6, 12, 32, 32})},
                {201, (64, 32, new[] {6, 12, 48, 32})}
            };

        public DenseNet(int num_init_features, int growth_rate, int[] block_config,
            int bn_size = 4, float? dropout = null, int classes = 1000,
            string prefix = "", ParameterDict @params = null) : base()
        {
            Features = new HybridSequential();
            Features.Add(new Conv2D(num_init_features, (7, 7), (2, 2), (3, 3), use_bias: false));
            Features.Add(new _BatchNorm());
            Features.Add(new Activation(ActivationType.Relu));
            Features.Add(new MaxPool2D((3, 3), (2, 2), (1, 1)));

            var num_features = num_init_features;
            for (var i = 0; i < block_config.Length; i++)
            {
                var num_layers = block_config[i];
                Features.Add(MakeDenseBlock(num_layers, bn_size, growth_rate, dropout, i + 1));
                num_features = num_features + num_layers * growth_rate;
                if (i != block_config.Length - 1)
                {
                    num_features = (int)Math.Truncate(Convert.ToDouble(num_features) / 2);
                    Features.Add(MakeTransition(num_features));
                }
            }

            Features.Add(new _BatchNorm());
            Features.Add(new Activation(ActivationType.Relu));
            Features.Add(new AvgPool2D((7, 7)));
            Features.Add(new Flatten());

            RegisterChild(Features, "features");

            Output = new Dense(classes);

            RegisterChild(Output, "output");
        }

        public HybridSequential Features { get; set; }
        public Dense Output { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            x = Features.Call(x, args);
            x = Output.Call(x, args);
            return x;
        }

        public static DenseNet GetDenseNet(int num_layers, bool pretrained = false, Context ctx = null,
            string root = "", int bn_size = 4, float? dropout = null, int classes = 1000)
        {
            var (num_init_features, growth_rate, block_config) = densenet_spec[num_layers];
            var net = new DenseNet(num_init_features, growth_rate, block_config, bn_size, dropout, classes);
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile("densenet" + num_layers, root), ctx);

            return net;
        }

        public static DenseNet DenseNet121(bool pretrained = false, Context ctx = null, string root = "",
            int bn_size = 4, float? dropout = null, int classes = 1000)
        {
            return GetDenseNet(121, pretrained, ctx, root, bn_size, dropout, classes);
        }

        public static DenseNet DenseNet161(bool pretrained = false, Context ctx = null, string root = "",
            int bn_size = 4, float? dropout = null, int classes = 1000)
        {
            return GetDenseNet(161, pretrained, ctx, root, bn_size, dropout, classes);
        }

        public static DenseNet DenseNet169(bool pretrained = false, Context ctx = null, string root = "",
            int bn_size = 4, float? dropout = null, int classes = 1000)
        {
            return GetDenseNet(169, pretrained, ctx, root, bn_size, dropout, classes);
        }

        public static DenseNet DenseNet201(bool pretrained = false, Context ctx = null, string root = "",
            int bn_size = 4, float? dropout = null, int classes = 1000)
        {
            return GetDenseNet(201, pretrained, ctx, root, bn_size, dropout, classes);
        }

        private static HybridSequential MakeDenseBlock(int num_layers, int bn_size, int growth_rate, float? dropout,
            int stage_index)
        {
            var block = new HybridSequential();
            for (var i = 0; i < num_layers; i++) block.Add(MakeDenseLayer(growth_rate, bn_size, dropout));

            return block;
        }

        private static HybridConcatenate MakeDenseLayer(int growth_rate, int bn_size, float? dropout)
        {
            var new_features = new HybridSequential();
            new_features.Add(new _BatchNorm());
            new_features.Add(new Activation(ActivationType.Relu));
            new_features.Add(new Conv2D(bn_size * growth_rate, (1, 1), use_bias: false));
            new_features.Add(new _BatchNorm());
            new_features.Add(new Activation(ActivationType.Relu));
            new_features.Add(new Conv2D(growth_rate, (3, 3), padding: (1, 1), use_bias: false));

            if (dropout.HasValue)
                new_features.Add(new Dropout(dropout.Value));

            var result = new HybridConcatenate(1);
            result.Add(new Identity());
            result.Add(new_features);

            return result;
        }

        private static HybridSequential MakeTransition(int num_output_features)
        {
            var block = new HybridSequential();
            block.Add(new _BatchNorm());
            block.Add(new Activation(ActivationType.Relu));
            block.Add(new Conv2D(num_output_features, (1, 1), use_bias: false));
            block.Add(new AvgPool2D((2, 2), (2, 2)));

            return block;
        }
    }
}