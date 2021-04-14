using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class UpSampling : BaseLayer
    {
        public int Scale { get; set; }

        public UpSampling(int scale = 2)
            :base("upsampling")
        {
            Scale = scale;
        }

        public override Symbol Build(Symbol x)
        {
            return new Operator("UpSampling").SetInput("data", x).SetParam("scale", Scale).CreateSymbol(ID);
        }
    }
}
