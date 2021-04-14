using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class Permute : BaseLayer
    {
        /// <summary>
        /// 
        /// </summary>
        public Permute()
            :base("permute")
        {
        }

        public override Symbol Build(Symbol data)
        {
            return sym.Transpose(data);
        }
        
    }
}
