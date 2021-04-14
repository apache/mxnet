using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Weibull : TransformedDistribution
    {
        public override NDArrayOrSymbol Mean
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public override NDArrayOrSymbol Variance
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public Weibull(Shape concentration, NDArrayOrSymbol scale = null, bool? validate_args = null)
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

        public override NDArrayOrSymbol Entropy()
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
