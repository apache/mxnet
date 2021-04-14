using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class LessThanEq : Constraint
    {
        public float _upper_bound;
        public LessThanEq(float upper_bound)
        {
            this._upper_bound = upper_bound;
        }

        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            var err_msg = $"Constraint violated: value should be less than equal {_upper_bound}";
            var condition = value <= this._upper_bound;
            var constraint_check = DistributionsUtils.ConstraintCheck();
            var _value = constraint_check(condition, err_msg) * value;
            return _value;
        }
    }
}
