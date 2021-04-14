using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class HalfOpenInterval : Constraint
    {
        public float _lower_bound;

        public float _upper_bound;

        public HalfOpenInterval(float lower_bound, float upper_bound)
        {
            this._lower_bound = lower_bound;
            this._upper_bound = upper_bound;
        }

        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            //var err_msg = $"Constraint violated: value should be >= {_lower_bound} and < {_upper_bound}.";
            //NDArrayOrSymbol condition = value.IsNDArray ? np.bitwise_and(value >= this._lower_bound, value < this._upper_bound)
            //                    : sym.LogicalAnd(value >= this._lower_bound, value < this._upper_bound);
            //var constraint_check = DistributionsUtils.ConstraintCheck();
            //var _value = constraint_check(condition, err_msg) * value;
            //return _value;

            throw new NotImplementedRelease1Exception();
        }
    }
}
