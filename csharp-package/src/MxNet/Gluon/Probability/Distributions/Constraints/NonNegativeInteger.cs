using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class NonNegativeInteger : IntegerGreaterThanEq
    {
        public NonNegativeInteger() : base(0)
        {

        }
    }
}
