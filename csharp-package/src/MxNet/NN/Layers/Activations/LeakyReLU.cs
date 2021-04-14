using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class LeakyReLU : BaseLayer
    {
        public float Alpha { get; set; }

        public LeakyReLU(float alpha=0.3f)
            : base("leakyrelu")
        {
            Alpha = alpha;
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "leaky")
                                            .SetInput("data", x)
                                            .SetParam("slope", Alpha)
                                            .CreateSymbol(ID);
        }
    }
}
