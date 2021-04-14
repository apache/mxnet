using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class Boolean : Constraint
    {
        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            var err_msg = "Constraint violated: value should be either 0 or 1.";
            NDArrayOrSymbol condition = value.IsNDArray ? nd.LesserEqual(value, 1) : sym.LesserEqual(value, sym.OnesLike(value));
            var constraint_check = DistributionsUtils.ConstraintCheck();
            var _value = constraint_check(condition, err_msg) * value;
            return _value;
        }
    }
}
