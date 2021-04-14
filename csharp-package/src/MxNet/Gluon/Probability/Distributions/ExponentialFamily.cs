using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class ExponentialFamily : Distribution
    {
        public ExponentialFamily(int? event_dim = null, bool? validate_args = null) : base(event_dim, validate_args)
        {
        }

        public virtual NDArrayOrSymbolList NaturalParams
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public virtual NDArrayOrSymbol LogNormalizer(NDArrayOrSymbol x)
        {
            throw new NotSupportedException();
        }

        public virtual NDArrayOrSymbol MeanCarrierMeasure(NDArrayOrSymbol x)
        {
            throw new NotSupportedException();
        }
    }
}
