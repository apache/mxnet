using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public abstract class Constraint
    {
        public abstract NDArrayOrSymbol Check(NDArrayOrSymbol value);

        public static bool IsDependent(Constraint constraint)
        {
            return constraint.GetType().Name == "_Dependent";
        }
    }
}
