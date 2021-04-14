using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions.Constraints
{
    public class Stack : Constraint
    {
        public Stack(NDArrayOrSymbol constraint_seq, int axis = 0)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol Check(NDArrayOrSymbol value)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
