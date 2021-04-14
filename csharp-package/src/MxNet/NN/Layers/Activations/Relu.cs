using MxNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Layers.Activations
{
    public class ReLU : BaseLayer
    {
        public ReLU()
            : base("relu")
        {

        }

        public override Symbol Build(Symbol x)
        {
            return sym.Relu(x, ID);
        }
    }
}
