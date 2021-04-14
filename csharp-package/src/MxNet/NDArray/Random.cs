/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
using MxNet.Interop;
using System;

namespace MxNet
{
    public partial class nd
    {
        public class Random
        {
            public static NDArray Uniform(float low = 0f, float high = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("random_uniform")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray Normal(float loc = 0f, float scale = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("random_normal")
                    .SetParam("loc", loc)
                    .SetParam("scale", scale)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray Gamma(float alpha = 1f, float beta = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("gamma")
                    .SetParam("alpha", alpha)
                    .SetParam("beta", beta)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray Exponential(float lam = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("exponential")
                    .SetParam("lam", lam)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray Poisson(float lam = 1f, Shape shape = null, Context ctx = null, DType dtype = null)
            {
                return new Operator("poisson")
                    .SetParam("lam", lam)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray NegativeBinomial(int k = 1, float p = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("negative_binomial")
                    .SetParam("k", k)
                    .SetParam("p", p)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray GeneralizedNegativeBinomial(float mu = 1f, float alpha = 1f, Shape shape = null,
                Context ctx = null, DType dtype = null)
            {
                return new Operator("generalized_negative_binomial")
                    .SetParam("mu", mu)
                    .SetParam("alpha", alpha)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray Randint(Tuple<double> low, Tuple<double> high, Shape shape = null, Context ctx = null,
                DType dtype = null)
            {
                return new Operator("randint")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype)
                    .Invoke();
            }

            public static NDArray UniformLike(NDArray data, float low = 0f, float high = 1f)
            {
                return new Operator("uniform_like")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray NormalLike(NDArray data, float loc = 0f, float scale = 1f)
            {
                return new Operator("normal_like")
                    .SetParam("loc", loc)
                    .SetParam("scale", scale)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray GammaLike(NDArray data, float alpha = 1f, float beta = 1f)
            {
                return new Operator("gamma_like")
                    .SetParam("alpha", alpha)
                    .SetParam("beta", beta)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray ExponentialLike(NDArray data, float lam = 1f)
            {
                return new Operator("exponential_like")
                    .SetParam("lam", lam)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray PoissonLike(NDArray data, float lam = 1f)
            {
                return new Operator("poisson_like")
                    .SetParam("lam", lam)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray NegativeBinomialLike(NDArray data, int k = 1, float p = 1f)
            {
                return new Operator("negative_binomial_like")
                    .SetParam("k", k)
                    .SetParam("p", p)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static NDArray GeneralizedNegativeBinomialLike(NDArray data, float mu = 1f, float alpha = 1f)
            {
                return new Operator("generalized_negative_binomial_like")
                    .SetParam("mu", mu)
                    .SetParam("alpha", alpha)
                    .SetInput("data", data)
                    .Invoke();
            }

            public static void Seed(int seed, Context ctx = null)
            {
                if (ctx == null)
                    NativeMethods.MXRandomSeed(seed);
                else
                    NativeMethods.MXRandomSeedContext(seed, (int)ctx.GetDeviceType(), ctx.GetDeviceId());
            }
        }
    }
}