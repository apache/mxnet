using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.NN.Initializers
{
    public class Zeros : Constant
    {
        public Zeros()
            : base(0f)
        {
            Name = "zeros";
        }
    }
}
