using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Softplus : BaseLayer
    {
        public Softplus()
            : base("softplus")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "softrelu")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
