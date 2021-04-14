using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Exp : BaseLayer
    {
        public Exp()
            : base("exp")
        {
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("exp").SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
