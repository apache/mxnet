using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class RelaxedOneHotCategorical : TransformedDistribution
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

        public RelaxedOneHotCategorical(int num_events = 1, NDArrayOrSymbol prob = null, NDArrayOrSymbol logit = null, bool? validate_args = null)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
