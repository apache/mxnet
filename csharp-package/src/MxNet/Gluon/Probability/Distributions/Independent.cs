using MxNet.Gluon.Probability.Distributions.Constraints;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Independent : Distribution
    {
        public bool HasEnumerateSupport
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public override Constraint Support
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

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

        public Independent(Distribution base_distribution, int reinterpreted_batch_ndims, bool? validate_args = null)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override Distribution BroadcastTo(Shape batch_shape)
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

        public override NDArrayOrSymbol LogProb(NDArrayOrSymbol value)
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol Entropy()
        {
            throw new NotImplementedRelease1Exception();
        }

        public override NDArrayOrSymbol EnumerateSupport()
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
