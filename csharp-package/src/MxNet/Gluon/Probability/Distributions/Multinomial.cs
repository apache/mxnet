using MxNet.Gluon.Probability.Distributions.Constraints;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Multinomial : Distribution
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

        public override Constraint Support => throw new NotImplementedRelease1Exception();

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

        public Multinomial(int num_events = 1, NDArrayOrSymbol prob = null, NDArrayOrSymbol logit = null, int total_count = 1, bool? validate_args = null)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override Distribution BroadcastTo(Shape batch_shape)
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
