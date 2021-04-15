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

namespace MxNet.Gluon.ModelZoo.Vision
{
    public class AlexNet : HybridBlock
    {
        public AlexNet(int classes = 1000) : base()
        {
            Features = new HybridSequential();
            Features.Add(new Conv2D(64, (11, 11), (4, 4), (2, 2), activation: ActivationType.Relu));
            Features.Add(new MaxPool2D((3, 3), (2, 2)));

            Features.Add(new Conv2D(192, (5, 5), padding: (2, 2), activation: ActivationType.Relu));
            Features.Add(new MaxPool2D((3, 3), (2, 2)));

            Features.Add(new Conv2D(384, (3, 3), padding: (1, 1), activation: ActivationType.Relu));
            Features.Add(new Conv2D(256, (3, 3), padding: (1, 1), activation: ActivationType.Relu));
            Features.Add(new Conv2D(256, (3, 3), padding: (1, 1), activation: ActivationType.Relu));
            Features.Add(new MaxPool2D((3, 3), (2, 2)));
            Features.Add(new Flatten());
            Features.Add(new Dense(4096, ActivationType.Relu));
            Features.Add(new Dropout(0.5f));
            Features.Add(new Dense(4096, ActivationType.Relu));
            Features.Add(new Dropout(0.5f));

            Output = new Dense(classes);

            RegisterChild(Features, "");
            RegisterChild(Output, "");
        }

        public HybridSequential Features { get; set; }
        public Dense Output { get; set; }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            args = Features.Call(args);
            args = Output.Call(args);
            return args;
        }

        public static AlexNet GetAlexNet(bool pretrained = false, Context ctx = null, string root = "")
        {
            var net = new AlexNet();
            if (ctx == null)
                ctx = Context.CurrentContext;
            if (pretrained) net.LoadParameters(ModelStore.GetModelFile("alexnet", root), ctx);

            return net;
        }
    }
}