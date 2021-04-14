using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Layers
{
    public class Reshape : BaseLayer
    {
        /// <summary>
        /// Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
        /// </summary>
        public Shape TargetShape { get; set; }

        /// <summary>
        ///  If true then the special values are inferred from right to left
        /// </summary>
        public bool Reverse { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targetShape"></param>
        /// <param name="reverse"></param>
        public Reshape(Shape targetShape, bool reverse = false)
            :base("reshape")
        {
            TargetShape = targetShape;
            Reverse = reverse;
        }

        public override Symbol Build(Symbol data)
        {
            return sym.Reshape(data, TargetShape, Reverse, ID);
        }
        
    }
}
