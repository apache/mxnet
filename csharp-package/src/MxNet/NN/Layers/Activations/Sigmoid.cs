using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Sigmoid : BaseLayer
    {
        public Sigmoid()
            : base("sigmoid")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "sigmoid")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
