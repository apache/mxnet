using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.Probability.Distributions
{
    public class DistributionsUtils
    {
        public static Func<NDArrayOrSymbol, string, NDArrayOrSymbol> ConstraintCheck()
        {
            throw new NotImplementedRelease1Exception();
        }

        public static Func<NDArrayOrSymbol, NDArrayOrSymbol> DiGamma()
        {
            throw new NotImplementedRelease1Exception();
        }

        public static Func<NDArrayOrSymbol, NDArrayOrSymbol> GammaLn()
        {
            throw new NotImplementedRelease1Exception();
        }

        public static Func<NDArrayOrSymbol, NDArrayOrSymbol> Erf()
        {
            throw new NotImplementedRelease1Exception();
        }

        public static Func<NDArrayOrSymbol, NDArrayOrSymbol> ErfInv()
        {
            throw new NotImplementedRelease1Exception();
        }

        public static Shape SampleNShapeConverter(Shape size)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static object GetF(NDArrayOrSymbolList @params)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static ndarray SumRightMost(NDArrayOrSymbol x, int ndim)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static NDArrayOrSymbol ClipProb(NDArrayOrSymbol prob)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static NDArrayOrSymbol ClipFloatEps(NDArrayOrSymbol value)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static NDArrayOrSymbol Prob2Logit(NDArrayOrSymbol prob, bool binary = true)
        {
            throw new NotImplementedRelease1Exception();
        }

        public static NDArrayOrSymbol Logit2Prob(NDArrayOrSymbol logit, bool binary = true)
        {
            throw new NotImplementedRelease1Exception();
        }
    }
}
