using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class NonNegative : GreaterThanEq
    {
        public NonNegative() : base(0)
        {

        }
    }
}
