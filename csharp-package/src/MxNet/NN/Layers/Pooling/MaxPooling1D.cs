using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class MaxPooling1D : BaseLayer
    {
        public uint PoolSize { get; set; }

        public uint Strides { get; set; }

        public uint? Padding { get; set; }

        public MaxPooling1D(uint poolSize = 2, uint? strides = null, uint? padding = null)
            :base("maxpooling1d")
        {
            PoolSize = poolSize;
            Strides = strides.HasValue ? padding.Value : poolSize;
            Padding = padding;
        }

        public override Symbol Build(Symbol x)
        {
            Shape pad = new Shape(); ;
            if (Padding.HasValue)
            {
                pad = new Shape(Padding.Value);
            }

            return sym.Pooling(x, new Shape(PoolSize), PoolingPoolType.Max, false, false, 
                                    PoolingPoolingConvention.Valid, new Shape(Strides), pad, 0, true, null, ID);
        }
    }
}
