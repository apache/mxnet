using MxNet.Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.ND.Numpy
{
    internal partial class Random
    {
        private static dynamic _api_internal = new _api_internals();

        public ndarray bernoulli(float? prob = null, float? logit = null, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            if ((prob == null) && (logit == null))
            {
                throw new Exception($"Either `prob` or `logit` must be specified, but not both. " + "Received prob={prob}, logit={logit}");
            }
            if (dtype == null)
            {
                dtype = np.Float32;
            }

            if (ctx == null)
            {
                ctx = Context.CurrentContext;
            }
            
            if (prob != null)
            {
                return _api_internal.bernoulli(prob: prob, logit: null, is_logit: false, size: size, ctx: ctx, dtype: dtype);
            }
            else
            {
                return _api_internal.bernoulli(prob: null, logit: logit, is_logit: true, size: size, ctx: ctx, dtype: dtype);
            }
        }

        public ndarray bernoulli(ndarray prob = null, ndarray logit = null, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            if ((prob == null) && (logit == null))
            {
                throw new Exception($"Either `prob` or `logit` must be specified, but not both. " + "Received prob={prob}, logit={logit}");
            }
            if (dtype == null)
            {
                dtype = np.Float32;
            }

            if (ctx == null)
            {
                ctx = Context.CurrentContext;
            }

            if (prob != null)
            {
                return _api_internal.bernoulli(data: prob, prob: null, logit: null, is_logit: false, size: size, ctx: ctx, dtype: dtype);
            }
            else
            {
                return _api_internal.bernoulli(data: logit, prob: null, logit: null, is_logit: true, size: size, ctx: ctx, dtype: dtype);
            }
        }

        public ndarray uniform_n(float low = 0, float high = 1, Shape batch_shape = null, DType dtype = null, Context ctx = null)
        {
            if (dtype == null)
            {
                dtype = np.Float32;
            }
            if (ctx == null)
            {
                ctx = Context.CurrentContext;
            }

            return _api_internal.uniform(low: low, high: high, size: batch_shape, ctx: ctx, dtype: dtype);
        }

        public ndarray normal_n(float loc = 0, float scale = 1, Shape batch_shape = null, DType dtype = null, Context ctx = null)
        {
            if (dtype == null)
            {
                dtype = np.Float32;
            }
            if (ctx == null)
            {
                ctx = Context.CurrentContext;
            }

            return _api_internal.normal(loc: loc, scale: scale, size: batch_shape, ctx: ctx, dtype: dtype);
        }
    }
}
