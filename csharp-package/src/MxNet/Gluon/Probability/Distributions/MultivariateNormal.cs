using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class MultivariateNormal : Distribution
    {
        public NDArrayOrSymbol ScaleTril
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public NDArrayOrSymbol Cov
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public NDArrayOrSymbol Precision
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public MultivariateNormal(NDArrayOrSymbol loc, NDArrayOrSymbol cov = null, NDArrayOrSymbol precision = null, NDArrayOrSymbol scale_tril = null, bool? validate_args = null)
        {
            throw new NotImplementedRelease1Exception();
        }

        private NDArrayOrSymbol _precision_to_scale_tril(NDArrayOrSymbol P)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol LogProb(NDArrayOrSymbol value)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol Sample(Shape size)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol SampleN(Shape size)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
