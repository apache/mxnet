using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class Chi2 : Gamma
    {
        public NDArrayOrSymbol Df
        {
            get
            {
                throw new NotImplementedRelease1Exception();
            }
        }

        public Chi2(NDArrayOrSymbol df, bool? validate_args = null) : base(df / 2, 2, validate_args)
        {

        }
    }
}
