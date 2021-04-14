using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class AvgPooling2D : BaseLayer
    {
        public Tuple<uint, uint> PoolSize { get; set; }

        public Tuple<uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public AvgPooling2D(Tuple<uint, uint> poolSize = null, Tuple<uint, uint> strides = null, uint? padding = null)
            :base("avgpooling2d")
        {
            PoolSize = poolSize ?? Tuple.Create<uint, uint>(2, 2);
            Strides = strides ?? poolSize;
            Padding = padding;
        }

        public override Symbol Build(Symbol x)
        {
            Shape pad = new Shape(); ;
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value);
            }

            return sym.Pooling(x, new Shape(PoolSize.Item1, PoolSize.Item2), PoolingPoolType.Avg, false, false, 
                                    PoolingPoolingConvention.Valid, new Shape(Strides.Item1, Strides.Item2), pad, 0, true, null, ID);
        }
    }
}
