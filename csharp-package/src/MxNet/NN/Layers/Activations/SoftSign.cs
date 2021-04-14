using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class SoftSign : BaseLayer
    {
        public SoftSign()
            : base("softsign")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("Activation").SetParam("act_type", "softsign")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
