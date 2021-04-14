using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Tanh : BaseLayer
    {
        public Tanh()
            : base("tanh")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "tanh")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
