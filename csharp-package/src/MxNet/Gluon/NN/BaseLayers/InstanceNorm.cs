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
using MxNet.Initializers;

namespace MxNet.Gluon.NN
{
    public class InstanceNorm : HybridBlock
    {
        public InstanceNorm(int axis = 1, float epsilon = 1e-5f, bool center = true, bool scale = false,
            string beta_initializer = "zeros", string gamma_initializer = "ones",
            int in_channels = 0) : base()
        {
            Axis = axis;
            Epsilon = epsilon;
            Center = center;
            Scale = scale;
            In_Channels = in_channels;
            this["gamma"] = Params.Get("gamma", scale ? OpGradReq.Write : OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(gamma_initializer), allow_deferred_init: true);
            this["beta"] = Params.Get("beta", center ? OpGradReq.Write : OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(beta_initializer), allow_deferred_init: true);
        }

        public int Axis { get; }
        public float Epsilon { get; }
        public bool Center { get; }
        public bool Scale { get; }
        public int In_Channels { get; }
        public Parameter Gamma { get; set; }
        public Parameter Beta { get; set; }

        public override NDArrayOrSymbol HybridForward(NDArrayOrSymbol x, NDArrayOrSymbolList args)
        {
            var gamma = args[0];
            var beta = args[1];

            if (Axis == 1)
            {
                if (x.IsNDArray)
                    return nd.InstanceNorm(x.NdX, gamma.NdX, beta.NdX, Epsilon);
                return sym.InstanceNorm(x.SymX, gamma.SymX, beta.SymX, Epsilon, "fwd");
            }

            if (x.IsNDArray)
            {
                var xs = nd.SwapAxis(x.NdX, 1, (uint)Axis);
                return nd.SwapAxis(nd.InstanceNorm(xs, gamma.NdX, beta.NdX, Epsilon), 1, (uint)Axis);
            }
            else
            {
                var xs = sym.SwapAxis(x.SymX, 1,(uint) Axis);
                return sym.SwapAxis(sym.InstanceNorm(xs, gamma.SymX, beta.SymX, Epsilon, "fwd"), 1, (uint)Axis);
            }
        }

        public override string ToString()
        {
            var in_channels = Params["gamma"].Shape[0];
            return $"{GetType().Name}(eps={Epsilon}, axis={Axis}, cneter={Center}, scale={Scale}, in_channels={in_channels})";
        }
    }
}