using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class GlobalPooling1D : BaseLayer
    {
        public PoolingPoolType PoolingType { get; set; }

        public GlobalPooling1D(PoolingPoolType poolingType)
            :base("globalpooling1d")
        {
            PoolingType = poolingType;
        }

        public override Symbol Build(Symbol x)
        {
            return sym.Pooling(x, new Shape(), PoolingType, true, false, 
                                    PoolingPoolingConvention.Valid, new Shape(), new Shape(), 0, true, null, ID);
        }
    }
}
