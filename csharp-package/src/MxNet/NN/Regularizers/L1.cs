using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Regularizers
{
    public class L1 : L1L2
    {
        public L1(float l=0.01f)
            : base(l, 0)
        {

        }
    }
}
