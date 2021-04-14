using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class AvgPooling3D : BaseLayer
    {
        public Tuple<uint, uint, uint> PoolSize { get; set; }

        public Tuple<uint, uint, uint> Strides { get; set; }

        public uint? Padding { get; set; }

        public AvgPooling3D(Tuple<uint, uint, uint> poolSize = null, Tuple<uint, uint, uint> strides = null, uint? padding = null)
            :base("avgpooling3d")
        {
            PoolSize = poolSize ?? Tuple.Create<uint, uint, uint>(2, 2, 2);
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

            return sym.Pooling(x, new Shape(PoolSize.Item1, PoolSize.Item2, PoolSize.Item3), PoolingPoolType.Avg, false, false, 
                                    PoolingPoolingConvention.Valid, new Shape(Strides.Item1, Strides.Item2, Strides.Item3), pad, 0, true, null, ID);
        }
    }
}
