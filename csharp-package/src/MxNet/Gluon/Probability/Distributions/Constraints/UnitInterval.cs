using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class UnitInterval : Interval
    {
        public UnitInterval() : base(0, 1)
        {

        }
    }
}
