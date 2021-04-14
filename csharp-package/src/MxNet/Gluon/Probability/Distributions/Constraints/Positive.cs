using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class Positive : GreaterThan
    {
        public Positive() : base(0)
        {

        }
    }
}
