using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Sym.Numpy
{
    internal partial class Random
    {
        public _Symbol randint(int low, int? high = null, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            if (high == null)
            {
                high = low;
                low = 0;
            }

            @out = _api_internal.random_randint(low: low, high: high, size: size, dtype: dtype, ctx: ctx);
            return @out;
        }

        public _Symbol uniform(float low = 0, float high = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.uniform(low: low, high: high, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public _Symbol normal(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.normal(loc: loc, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public _Symbol lognormal(float mean = 0, float sigma = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            return sym_np_ops.exp(normal(loc: mean, scale: sigma, size: size, dtype: dtype, ctx: ctx, @out: @out));
        }

        public _Symbol logistic(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.logistic(loc: loc, scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol gumbel(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.gumbel(loc: loc, scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol multinomial(int n, float[] pvals, Shape size = null)
        {
            return _api_internal.multinomial(n: n, pvals: pvals, size: size);
        }

        public _Symbol multivariate_normal(_Symbol mean, _Symbol cov, Shape size = null, string check_valid = null, float? tol = null)
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

        public _Symbol choice(_Symbol a, Shape size = null, bool replace = true, _Symbol p = null, Context ctx = null, _Symbol @out = null)
        {
            var indices = _api_internal.choice(a: a, size: size, replace: replace, p: p, ctx: ctx);
            @out = _api_internal.take(a, indices, 0, "raise");
            return @out;
        }

        public _Symbol choice(int a, Shape size = null, bool replace = true, _Symbol p = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.choice(a: a, size: size, replace: replace, p: p, ctx: ctx);
            return @out;
        }

        public _Symbol rayleigh(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.rayleigh(scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol rand(Shape size = null)
        {
            return uniform(size: size);
        }

        public _Symbol exponential(float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.exponential(scale: scale, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol weibull(float a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.weibull(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol weibull(_Symbol a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.weibull(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol pareto(float a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.pareto(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol pareto(_Symbol a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.pareto(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol power(float a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.powerd(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol power(_Symbol a, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.powerd(a: a, size: size, ctx: ctx);
            return @out;
        }

        public _Symbol shuffle(_Symbol x)
        {
            return _api_internal.shuffle(x, x);
        }

        public _Symbol gamma(float shape, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.gamma(shape: shape, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public _Symbol gamma(_Symbol shape, float scale, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.gamma(input1: shape, shape: null, scale: scale, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public _Symbol gamma(_Symbol shape, _Symbol scale, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.gamma(input1: shape, input2: scale, shape: null, scale: null, size: size, ctx: ctx, dtype: dtype);
            return @out;
        }

        public _Symbol beta(float a, float b, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            var X = gamma(a, 1, size: size, dtype: "float64", ctx: ctx);
            var Y = gamma(b, 1, size: size, dtype: "float64", ctx: ctx);
            @out = X / (X + Y);
            return @out;
        }

        public _Symbol beta(_Symbol a, _Symbol b, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            var X = gamma(a, 1, size: size, dtype: "float64", ctx: ctx);
            var Y = gamma(b, 1, size: size, dtype: "float64", ctx: ctx);
            @out = X / (X + Y);
            return @out;
        }

        public _Symbol f(float dfnum, float dfden, Shape size = null, Context ctx = null, _Symbol @out = null)
        {
            var X = chisquare(df: dfnum, size: size, ctx: ctx);
            var Y = chisquare(df: dfden, size: size, ctx: ctx);
            return X * dfden / (Y * dfnum);
        }

        public _Symbol f(_Symbol dfnum, _Symbol dfden, Shape size = null, Context ctx = null, _Symbol @out = null)
        {
            var X = chisquare(df: dfnum, size: size, ctx: ctx);
            var Y = chisquare(df: dfden, size: size, ctx: ctx);
            return X * dfden / (Y * dfnum);
        }

        public _Symbol chisquare(float df, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            return gamma(df / 2, 2, size: size, dtype: dtype, ctx: ctx);
        }

        public _Symbol chisquare(_Symbol df, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            return gamma(df / 2, 2, size: size, dtype: dtype, ctx: ctx);
        }

        public _Symbol randn(params int[] size)
        {
            return uniform(0, 1, size: new Shape(size));
        }

        public _Symbol laplace(float loc = 0, float scale = 1, Shape size = null, DType dtype = null, Context ctx = null, _Symbol @out = null)
        {
            @out = _api_internal.laplace(loc: loc, scale: scale, size: size, dtype: dtype, ctx: ctx);
            return @out;
        }
    }
}
