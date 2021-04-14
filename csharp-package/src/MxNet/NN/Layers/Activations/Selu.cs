using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Selu : BaseLayer
    {
        public float Alpha { get; set; }

        public Selu(float alpha=1)
            : base("elu")
        {
            Alpha = alpha;
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "selu")
                                            .SetInput("data", x)
                                            .CreateSymbol(ID);
        }
    }
}
