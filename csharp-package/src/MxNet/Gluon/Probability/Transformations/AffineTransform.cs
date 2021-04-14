using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Transformations
{
    public class AffineTransform : Transformation
    {
        public AffineTransform(float loc, float scale, int event_dim = 0)
        {
            throw new NotImplementedRelease1Exception();
        }

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
            throw new NotImplementedRelease1Exception();
        }
    }
}
