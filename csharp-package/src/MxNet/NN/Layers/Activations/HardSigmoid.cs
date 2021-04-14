using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class HardSigmoid : BaseLayer
    {
        public HardSigmoid()
            : base("hardsigmoid")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("HardSigmoid").SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
