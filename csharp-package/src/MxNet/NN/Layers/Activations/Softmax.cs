using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Softmax : BaseLayer
    {
        public Softmax()
            : base("softmax")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("SoftmaxActivation").SetInput("data", x)
                                            .CreateSymbol();
        }
    }
}
