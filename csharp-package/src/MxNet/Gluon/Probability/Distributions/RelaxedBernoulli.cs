using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class RelaxedBernoulli : TransformedDistribution
    {
        public TransformedDistribution T
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public NDArrayOrSymbol Prob
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public NDArrayOrSymbol Logit
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public RelaxedBernoulli(NDArrayOrSymbol prob = null, NDArrayOrSymbol logit = null, bool? validate_args = null)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override Distribution BroadcastTo(Shape batch_shape)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
