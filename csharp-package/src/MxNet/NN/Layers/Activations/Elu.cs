using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class Elu : BaseLayer
    {
        public float Alpha { get; set; }

        public Elu(float alpha=1)
            : base("elu")
        {
            Alpha = alpha;
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "elu")
                                            .SetInput("data", x)
                                            .SetParam("slope", Alpha)
                                            .CreateSymbol(ID);
        }
    }
}
