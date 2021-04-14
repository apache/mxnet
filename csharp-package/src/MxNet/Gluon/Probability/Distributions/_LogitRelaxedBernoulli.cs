using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class _LogitRelaxedBernoulli : Distribution
    {
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

        public _LogitRelaxedBernoulli(NDArrayOrSymbol prob = null, NDArrayOrSymbol logit = null, bool? validate_args = null)
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
    }
}
