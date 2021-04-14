using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class PositiveInteger : IntegerGreaterThan
    {
        public PositiveInteger() : base(0)
        {

        }
    }
}
