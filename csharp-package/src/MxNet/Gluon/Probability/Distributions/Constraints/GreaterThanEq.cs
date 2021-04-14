using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class GreaterThanEq : Constraint
    {
        public float _lower_bound;
        public GreaterThanEq(float lower_bound)
        {
            this._lower_bound = lower_bound;
        }

        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            var err_msg = $"Constraint violated: value should be greater than equal {_lower_bound}";
            var condition = value >= this._lower_bound;
            var constraint_check = DistributionsUtils.ConstraintCheck();
            var _value = constraint_check(condition, err_msg) * value;
            return _value;
        }
    }
}
