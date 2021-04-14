using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Transformations
{
    public class AbsTransform : Transformation
    {
        public override NDArrayOrSymbol ForwardCompute(NDArrayOrSymbol x)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol InverseCompute(NDArrayOrSymbol x)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol LogDetJacobian(NDArrayOrSymbol x, NDArrayOrSymbol y)
        {
            throw new NotSupportedException();
        }
    }
}
