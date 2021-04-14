using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class IntegerInterval : Constraint
    {
        public int _lower_bound;

        public int _upper_bound;

        public IntegerInterval(int lower_bound, int upper_bound)
        {
            this._lower_bound = lower_bound;
            this._upper_bound = upper_bound;
        }

        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            var err_msg = $"Constraint violated: value should be >= {_lower_bound} and <= {_upper_bound}.";
            NDArrayOrSymbol condition = value.IsNDArray ? nd.EqualScalar(nd.Mod(value, 1), 0) : sym.EqualScalar(sym.Mod(value, sym.OnesLike(value)), 0);
            NDArrayOrSymbol condition1 = value.IsNDArray ? nd.LogicalAnd(value >= this._lower_bound, value <= this._upper_bound)
                                : sym.LogicalAnd(value >= this._lower_bound, value <= this._upper_bound);

            condition = nd.LogicalAnd(condition, condition1);

            var constraint_check = DistributionsUtils.ConstraintCheck();
            var _value = constraint_check(condition, err_msg) * value;
            return _value;
        }
    }
}
