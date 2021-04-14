using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class Flatten : BaseLayer
    {
        public Flatten()
            :base("flatten")
        {

        }

        public override Symbol Build(Symbol data)
        {
            return sym.Flatten(data, ID);
        }
        
    }
}
