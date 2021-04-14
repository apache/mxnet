using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class _Dependent : Constraint
    {
        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            throw new Exception("Cannot validate dependent constraint");
        }
    }
}
