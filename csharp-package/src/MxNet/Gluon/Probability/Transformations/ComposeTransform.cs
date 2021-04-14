using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Transformations
{
    public class ComposeTransform : Transformation
    {
        public override NDArrayOrSymbol F { get => throw new NotImplementedRelease1Exception(); set => throw new NotImplementedRelease1Exception(); }

        public override NDArrayOrSymbol Sign => throw new NotImplementedRelease1Exception();

        public override NDArrayOrSymbol Inv => throw new NotImplementedRelease1Exception();

        public int EventDim
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
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
