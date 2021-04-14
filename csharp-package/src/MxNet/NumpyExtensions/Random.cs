using MxNet.ND.Numpy;
using System;
using System.Collections.Generic;
using System.Text;
namespace MxNet.Numpy
{
    public partial class Random
    {
        public void seed(int seed, Context ctx = null)
        {
            mx.Seed(seed, ctx);
        }

        public ndarray bernoulli(float? prob = null, float? logit = null, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out= null)
        {
            return nd_np_ops.random.bernoulli(prob, logit, size, dtype, ctx, @out);
        }

        public ndarray bernoulli(ndarray prob = null, ndarray logit = null, Shape size = null, DType dtype = null, Context ctx = null, ndarray @out = null)
        {
            return nd_np_ops.random.bernoulli(prob, logit, size, dtype, ctx, @out);
        }

        public ndarray uniform_n(float low= 0, float high= 1, Shape batch_shape= null, DType dtype = null, Context ctx = null)
        {
            return nd_np_ops.random.uniform_n(low, high, batch_shape, dtype, ctx);
        }

        public ndarray normal_n(float loc = 0, float scale = 1, Shape batch_shape = null, DType dtype = null, Context ctx = null)
        {
            return nd_np_ops.random.normal_n(loc, scale, batch_shape, dtype, ctx);
        }
    }
}
