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

namespace MxNet
{
    public partial class sym
    {
        public class Random
        {
            public static Symbol Uniform(float low = 0f, float high = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("uniform")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol Normal(float loc = 0f, float scale = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("normal")
                    .SetParam("loc", loc)
                    .SetParam("scale", scale)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol Gamma(float alpha = 1f, float beta = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("gamma")
                    .SetParam("alpha", alpha)
                    .SetParam("beta", beta)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol Exponential(float lam = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("exponential")
                    .SetParam("lam", lam)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol Poisson(float lam = 1f, Shape shape = null, Context ctx = null, DType dtype = null, string name = "")
            {
                return new Operator("poisson")
                    .SetParam("lam", lam)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol NegativeBinomial(int k = 1, float p = 1f, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("negative_binomial")
                    .SetParam("k", k)
                    .SetParam("p", p)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol GeneralizedNegativeBinomial(float mu = 1f, float alpha = 1f, Shape shape = null,
                Context ctx = null, DType dtype = null, string name = "")
            {
                return new Operator("generalized_negative_binomial")
                    .SetParam("mu", mu)
                    .SetParam("alpha", alpha)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol Randint(Tuple<double> low, Tuple<double> high, Shape shape = null, Context ctx = null,
                DType dtype = null, string name = "")
            {
                return new Operator("randint")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetParam("shape", shape)
                    .SetParam("ctx", ctx)
                    .SetParam("dtype", dtype).CreateSymbol(name);
            }

            public static Symbol UniformLike(Symbol data, float low = 0f, float high = 1f, string name = "")
            {
                return new Operator("uniform_like")
                    .SetParam("low", low)
                    .SetParam("high", high)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol NormalLike(Symbol data, float loc = 0f, float scale = 1f, string name = "")
            {
                return new Operator("normal_like")
                    .SetParam("loc", loc)
                    .SetParam("scale", scale)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol GammaLike(Symbol data, float alpha = 1f, float beta = 1f, string name = "")
            {
                return new Operator("gamma_like")
                    .SetParam("alpha", alpha)
                    .SetParam("beta", beta)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol ExponentialLike(Symbol data, float lam = 1f, string name = "")
            {
                return new Operator("exponential_like")
                    .SetParam("lam", lam)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol PoissonLike(Symbol data, float lam = 1f, string name = "")
            {
                return new Operator("poisson_like")
                    .SetParam("lam", lam)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol NegativeBinomialLike(Symbol data, int k = 1, float p = 1f, string name = "")
            {
                return new Operator("negative_binomial_like")
                    .SetParam("k", k)
                    .SetParam("p", p)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static Symbol GeneralizedNegativeBinomialLike(Symbol data, float mu = 1f, float alpha = 1f, string name = "")
            {
                return new Operator("generalized_negative_binomial_like")
                    .SetParam("mu", mu)
                    .SetParam("alpha", alpha)
                    .SetInput("data", data).CreateSymbol(name);
            }

            public static void Seed(int seed, Context ctx = null)
            {
                if (ctx == null)
                    NativeMethods.MXRandomSeed(seed);
                else
                    NativeMethods.MXRandomSeedContext(seed, (int)ctx.GetDeviceType(), ctx.GetDeviceId());
            }

            public static Symbol Randn(float loc = 0, float scale = 1, DType dtype = null, Shape shape = null)
            {
                return RandomNormal(loc, scale, shape, dtype: dtype);
            }
        }
    }
}