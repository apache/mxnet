using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.ND.Numpy
{
    internal partial class Random
    {
        public ndarray randint(int low, int? high = null, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            if (high == null)
            {
                high = low;
                low = 0;
            }

            @out = _api_internal.random_randint(low: low, high: high, size: size, dtype: dtype, ctx: ctx);
            return @out;
        }

        public ndarray uniform(float low = 0, float high = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.uniform(low: low, high: high, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public ndarray normal(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.normal(loc: loc, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public ndarray lognormal(float mean = 0, float sigma = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.exp(normal(loc: mean, scale: sigma, size: size, dtype: dtype, ctx: ctx, @out: @out));
        }

        public ndarray logistic(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.logistic(loc: loc, scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public ndarray gumbel(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.gumbel(loc: loc, scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public ndarray multinomial(int n, float[] pvals, Shape size = null)
        {
            return _api_internal.multinomial(n: n, pvals: pvals, size: size);
        }

        public ndarray multivariate_normal(ndarray mean, ndarray cov, Shape size = null, string check_valid = null, float? tol = null)
        {
            if (check_valid != null)
            {
                throw new NotImplementedException("Parameter `check_valid` is not supported");
            }

            if (tol != null)
            {
                throw new NotImplementedException("Parameter `tol` is not supported");
            }

            return _api_internal.mvn_fallback(mean: mean, cov: cov, size: size);
        }

        public ndarray choice(ndarray a, Shape size = null, bool replace = true, ndarray p = null, Context ctx = null, ndarray @out = null)
        {
            var indices = _api_internal.choice(a: a, size: size, replace: replace, p: p, ctx: ctx);
            @out = _api_internal.take(a, indices, 0, "raise");
            return @out;
        }

        public ndarray choice(int a, Shape size = null, bool replace = true, ndarray p = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.choice(a: a, size: size, replace: replace, p: p, ctx: ctx);
            return @out;
        }

        public ndarray rayleigh(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.rayleigh(scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public ndarray rand(Shape size = null)
        {
            throw new NotImplementedException();
        }

        public ndarray exponential(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.exponential(scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public ndarray weibull(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.weibull(a: a, size: size, ctx: ctx);
            return @out;
        }

        public ndarray weibull(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.weibull(a: a, size: size, ctx: ctx);
            return @out;
        }

        public ndarray pareto(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.pareto(a: a, size: size, ctx: ctx);
            return @out;
        }

        public ndarray pareto(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.pareto(a: a, size: size, ctx: ctx);
            return @out;
        }

        public ndarray power(float a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.powerd(a: a, size: size, ctx:ctx);
            return @out;
        }

        public ndarray power(ndarray a, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.powerd(a: a, size: size, ctx: ctx);
            return @out;
        }

        public ndarray shuffle(ndarray x)
        {
            return _api_internal.shuffle(x, x);
        }

        public ndarray gamma(float shape, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.gamma(shape: shape, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public ndarray gamma(ndarray shape, ndarray scale, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.gamma(input1: shape, input2: scale, shape: null, scale: null, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public ndarray gamma(ndarray shape, float scale, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.gamma(input1: shape, shape: null, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public ndarray beta(float a, float b, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            var X = gamma(a, 1, size: size, dtype: "float64", ctx: ctx);
            var Y = gamma(b, 1, size: size, dtype: "float64", ctx: ctx);
            @out = X / (X + Y);
            return @out.AsType(dtype);
        }

        public ndarray beta(ndarray a, ndarray b, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            var X = gamma(a, 1f, size: size, dtype: "float64", ctx: ctx);
            var Y = gamma(b, 1f, size: size, dtype: "float64", ctx: ctx);
            @out = X / (X + Y);
            return @out.AsType(dtype);
        }

        public ndarray f(float dfnum, float dfden, Shape size = null, Context ctx = null, ndarray @out = null)
        {
            var X = chisquare(df: dfnum, size: size, ctx: ctx);
            var Y = chisquare(df: dfden, size: size, ctx: ctx);
            return X * dfden / (Y * dfnum);
        }

        public ndarray f(ndarray dfnum, ndarray dfden, Shape size = null, Context ctx = null, ndarray @out = null)
        {
            var X = chisquare(df: dfnum, size: size, ctx: ctx);
            var Y = chisquare(df: dfden, size: size, ctx: ctx);
            return X * dfden / (Y * dfnum);
        }

        public ndarray chisquare(float df, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return gamma(df / 2, 2f, size: size, dtype: dtype, ctx: ctx);
        }

        public ndarray chisquare(ndarray df, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return gamma(df / 2, 2f, size: size, dtype: dtype, ctx: ctx);
        }

        public ndarray randn(params int[] size)
        {
            return uniform(0, 1, size: new Shape(size));
        }

        public ndarray laplace(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            @out = _api_internal.laplace(loc: loc, scale: scale, size: size, dtype: dtype, ctx: ctx);
            return @out;
        }
    }
}
