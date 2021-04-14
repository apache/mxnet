using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class Real : Constraint
    {
        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            var err_msg = "Constraint violated: value should be a real tensor";

            // False when value has NANs
            NDArrayOrSymbol condition = value.IsNDArray ? nd.Equal(value, value) : sym.Equal(value, value);
            var constraint_check = DistributionsUtils.ConstraintCheck();
            var _value = constraint_check(condition, err_msg) * value;
            return _value;
        }
    }
}
