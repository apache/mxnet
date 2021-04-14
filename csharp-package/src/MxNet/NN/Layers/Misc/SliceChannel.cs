using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class SliceChannel : BaseLayer
    {
        public int NumOutputs { get; set; }

        public int Axis { get; set; }

        public bool Squeeze { get; set; }

        public SliceChannel(int numOutputs, int axis = 1, bool squeeze = false)
            : base("slicechannel")
        {
            NumOutputs = numOutputs;
            Axis = axis;
            Squeeze = squeeze;
        }

        public override Symbol Build(Symbol x)
        {
            return sym.SliceChannel(x, NumOutputs, Axis, Squeeze, ID);
        }
    }
}
