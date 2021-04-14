using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Numpy
{
    public partial class Random
    {
        public ndarray randint(int low, int? high= null, Shape size= null, DType dtype= null, Context ctx= null, ndarray @out= null)
        {
            return nd_np_ops.random.randint(low, high, size, dtype, ctx, @out);
        }

        public ndarray uniform(float low = 0, float high = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.uniform(low, high, size, dtype, ctx, @out);
        }

        public ndarray normal(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.normal(loc, scale, size, dtype, ctx, @out);
        }

        public ndarray lognormal(float mean = 0, float sigma = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.lognormal(mean, sigma, size, dtype, ctx, @out);
        }

        public ndarray logistic(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.logistic(loc, scale, size, dtype, ctx, @out);
        }

        public ndarray gumbel(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.gumbel(loc, scale, size, dtype, ctx, @out);
        }

        public ndarray multinomial(int n, float[] pvals, Shape size = null)
        {
            return nd_np_ops.random.multinomial(n, pvals, size);
        }

        public ndarray multivariate_normal(ndarray mean, ndarray cov, Shape size= null, string check_valid= null, float? tol= null)
        {
            return nd_np_ops.random.multivariate_normal(mean, cov, size, check_valid, tol);
        }

        public ndarray choice(ndarray a, Shape size= null, bool replace= true, ndarray p= null, Context ctx= null, ndarray @out= null)
        {
            return nd_np_ops.random.choice(a, size, replace, p, ctx, @out);
        }

        public ndarray rayleigh(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.rayleigh(scale, size, dtype, ctx, @out);
        }

        public ndarray rand(Shape size = null)
        {
            return nd_np_ops.random.rand(size);
        }

        public ndarray exponential(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.exponential(scale, size, dtype, ctx, @out);
        }

        public ndarray weibull(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.weibull(a, size, dtype, ctx, @out);
        }

        public ndarray weibull(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.weibull(a, size, dtype, ctx, @out);
        }

        public ndarray pareto(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.pareto(a, size, dtype, ctx, @out);
        }

        public ndarray pareto(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.pareto(a, size, dtype, ctx, @out);
        }

        public ndarray power(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.power(a, size, dtype, ctx, @out);
        }

        public ndarray power(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.power(a, size, dtype, ctx, @out);
        }

        public ndarray shuffle(ndarray x)
        {
            return nd_np_ops.random.shuffle(x);
        }

        public ndarray gamma(float shape, float scale=1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.gamma(shape, scale, size, dtype, ctx, @out);
        }

        public ndarray gamma(ndarray shape, ndarray scale, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.gamma(shape, scale, size, dtype, ctx, @out);
        }

        public ndarray beta(float a, float b, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.beta(a, b, size, dtype, ctx, @out);
        }

        public ndarray beta(ndarray a, ndarray b, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.beta(a, b, size, dtype, ctx, @out);
        }

        public ndarray f(float dfnum, float dfden, Shape size = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.f(dfnum, dfden, size, ctx, @out);
        }

        public ndarray f(ndarray dfnum, ndarray dfden, Shape size = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.f(dfnum, dfden, size, ctx, @out);
        }

        public ndarray chisquare(float df, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.chisquare(df, size, dtype, ctx, @out);
        }

        public ndarray chisquare(ndarray df, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.chisquare(df, size, dtype, ctx, @out);
        }

        public ndarray randn(params int[] size)
        {
            return nd_np_ops.random.randn(size);
        }

        public ndarray laplace(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.laplace(loc, scale, size, dtype, ctx, @out);
        }
    }
}
