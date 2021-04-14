using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class PReLU : BaseLayer
    {
        public PReLU()
            : base("prelu")
        {
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "prelu")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
