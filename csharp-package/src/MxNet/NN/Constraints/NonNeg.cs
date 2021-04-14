using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Constraints
{
    public class NonNeg : BaseConstraint
    {
        public NonNeg()
        {
        }

        public override NDArray Call(NDArray w)
        {
            w *= nd.Cast(nd.GreaterEqualScalar(w, 0), DType.Float32);
            return w;
        }
    }
}
