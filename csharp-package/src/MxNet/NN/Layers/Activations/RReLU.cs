using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class RReLU : BaseLayer
    {
        public float LowerBound { get; set; }

        public float UpperBound { get; set; }

        public RReLU(float lower_bound = 0.125f, float upper_bound=0.334f)
            : base("rrelu")
        {
            LowerBound = lower_bound;
            UpperBound = upper_bound;
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("LeakyReLU").SetParam("act_type", "rrelu")
                                            .SetInput("data", x)
                                            .SetParam("lower_bound", LowerBound)
                                            .SetParam("upper_bound", UpperBound)
                                            .CreateSymbol(ID);
        }
    }
}
