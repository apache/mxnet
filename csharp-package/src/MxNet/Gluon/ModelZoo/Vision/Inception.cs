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
using MxNet.Gluon.NN;
using System;


namespace MxNet.Gluon.ModelZoo.Vision
{
    using ConvSetting = Tuple<int, (int, int), (int, int)?, (int, int)?>;

    public class Inception
    {
        internal static ConvSetting MakeConvSetting(
            int channels, (int, int) kernel_size, (int, int)? strides = null, (int, int)? padding = null)
        {
            return Tuple.Create(channels, kernel_size, strides, padding);
        }

        internal static HybridSequential MakeBasicConv(int channels, (int, int) kernel_size,
            (int, int)? strides = default, (int, int)? padding = default)
        {
            var output = new HybridSequential();
            output.Add(new Conv2D(channels, kernel_size, strides, padding, use_bias: false));
            output.Add(new _BatchNorm(epsilon: 0.001f));
            output.Add(new Activation(ActivationType.Relu));

            return output;
        }

        internal static HybridSequential MakeBranch(string use_pool, ConvSetting[] conv_settings = null)
        {
            var output = new HybridSequential();
            if (use_pool == "avg")
                output.Add(new AvgPool2D((3, 3), (1, 1), (1, 1)));
            else if (use_pool == "max")
                output.Add(new MaxPool2D((3, 3), (2, 2)));

            if (conv_settings != null)
            {
                foreach (var cs in conv_settings)
                    output.Add(MakeBasicConv(cs.Item1, cs.Item2, cs.Item3, cs.Item4));
            }

            return output;
        }

        internal static HybridConcatenate MakeA(int pool_features)
        {
            var output = new HybridConcatenate(1);
            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(64, (1, 1))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(48, (1, 1)),
                    MakeConvSetting(64, (5, 5), null, (2, 2))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(64, (1, 1)),
                    MakeConvSetting(96, (3, 3), null, (1, 1)),
                    MakeConvSetting(96, (3, 3), null, (1, 1))
                }));

            output.Add(MakeBranch("avg", new[] {
                    MakeConvSetting(pool_features, (1, 1))
                }));

            return output;
        }

        internal static HybridConcatenate MakeB()
        {
            var output = new HybridConcatenate(1);
            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (3, 3), (2, 2))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(64, (1, 1)),
                    MakeConvSetting(96, (3, 3), null, (1, 1)),
                    MakeConvSetting(96, (3, 3), (2, 2))
                }));

            output.Add(MakeBranch("max"));

            return output;
        }

        internal static HybridConcatenate MakeC(int channels_7x7)
        {
            var output = new HybridConcatenate(1);
            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(192, (1, 1))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(channels_7x7, (1, 1)),
                    MakeConvSetting(channels_7x7, (1, 7), null, (0, 3)),
                    MakeConvSetting(192, (7, 1), null, (3, 0))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(channels_7x7, (1, 1)),
                    MakeConvSetting(channels_7x7, (7, 1), null, (3, 0)),
                    MakeConvSetting(channels_7x7, (1, 7), null, (0, 3)),
                    MakeConvSetting(channels_7x7, (7, 1), null, (3, 0)),
                    MakeConvSetting(192, (1, 7), null, (0, 3))
                }));

            output.Add(MakeBranch("avg", new[] {
                    MakeConvSetting(192, (1, 1))
                }));

            return output;
        }

        internal static HybridConcatenate MakeD()
        {
            var output = new HybridConcatenate(1);
            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(192, (1, 1)),
                    MakeConvSetting(320, (3, 3), (2, 2))
                }));

            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(192, (1, 1)),
                    MakeConvSetting(192, (1, 7), null, (0, 3)),
                    MakeConvSetting(192, (7, 1), null, (3, 0)),
                    MakeConvSetting(192, (3, 3), (2, 2))
                }));

            output.Add(MakeBranch("max"));

            return output;
        }

        internal static HybridConcatenate MakeE()
        {
            var output = new HybridConcatenate(1);
            output.Add(MakeBranch("", new[] {
                    MakeConvSetting(320, (1, 1))
                }));

            var branch_3x3 = new HybridSequential();
            branch_3x3.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (1, 1))
                }));
            output.Add(branch_3x3);

            var branch_3x3_split = new HybridConcatenate(1);
            branch_3x3_split.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (1, 3), null, (0, 1))
                }));
            branch_3x3_split.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (3, 1), null, (1, 0))
                }));
            branch_3x3.Add(branch_3x3_split);

            var branch_3x3dbl = new HybridSequential();
            branch_3x3dbl.Add(MakeBranch("", new[] {
                    MakeConvSetting(448, (1, 1)),
                    MakeConvSetting(384, (3, 3), null, (1, 1))
                }));
            output.Add(branch_3x3dbl);

            var branch_3x3dbl_split = new HybridConcatenate(1);
            branch_3x3dbl_split.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (1, 3), null, (0, 1))
                }));
            branch_3x3dbl_split.Add(MakeBranch("", new[] {
                    MakeConvSetting(384, (3, 1), null, (1, 0))
                }));
            branch_3x3dbl.Add(branch_3x3dbl_split);

            output.Add(MakeBranch("avg", new[] {
                    MakeConvSetting(192, (1, 1))
                }));
            return output;
        }

        internal static HybridSequential MakeAux(int classes)
        {
            var output = new HybridSequential();
            output.Add(new AvgPool2D((5, 5), (3, 3)));
            output.Add(MakeBasicConv(128, (1, 1)));
            output.Add(MakeBasicConv(768, (5, 5)));
            output.Add(new Flatten());
            output.Add(new Dense(classes));
            return output;
        }

        public static Inception3 GetInception3(bool pretrained = false, Context ctx = null, string root = "",
            int classes = 1000, string prefix = "", ParameterDict @params = null)
        {
            var net = new Inception3(classes);
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile("inceptionv3"), ctx);

            return net;
        }
    }

    public class Inception3 : HybridBlock
    {
        public Inception3(int classes = 1000) : base()
        {
            Features = new HybridSequential();
            Features.Add(Inception.MakeBasicConv(32, (3, 3), (2, 2)));
            Features.Add(Inception.MakeBasicConv(32, (3, 3)));
            Features.Add(Inception.MakeBasicConv(64, (3, 3), padding: (1, 1)));
            Features.Add(new MaxPool2D((3, 3), (2, 2)));
            Features.Add(Inception.MakeBasicConv(80, (1, 1)));
            Features.Add(Inception.MakeBasicConv(192, (3, 3)));
            Features.Add(new MaxPool2D((3, 3), (2, 2)));
            Features.Add(Inception.MakeA(32));
            Features.Add(Inception.MakeA(64));
            Features.Add(Inception.MakeA(64));
            Features.Add(Inception.MakeB());
            Features.Add(Inception.MakeC(128));
            Features.Add(Inception.MakeC(160));
            Features.Add(Inception.MakeC(160));
            Features.Add(Inception.MakeC(192));
            Features.Add(Inception.MakeD());
            Features.Add(Inception.MakeE());
            Features.Add(Inception.MakeE());
            Features.Add(new AvgPool2D((8, 8)));
            Features.Add(new Dropout(0.5f));
            RegisterChild(Features, "features");

            Output = new Dense(classes);
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
    }
}