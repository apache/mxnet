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
    public class _BatchNorm : HybridBlock
    {
        public _BatchNorm(int axis = 1, float momentum = 0.9f, float epsilon = 1e-5f, bool center = true, bool scale = true,
            bool fuse_relu = false, bool use_global_stats = false, string beta_initializer = "zeros", string gamma_initializer = "ones",
            string running_mean_initializer = "zeros", string running_variance_initializer = "ones",
            int in_channels = 0) : base()
        {
            Axis = axis;
            Momentum = momentum;
            Epsilon = epsilon;
            Center = center;
            Scale = scale;
            Use_Global_Stats = use_global_stats;
            In_Channels = in_channels;
            FuseRelu = fuse_relu;
            this["gamma"] = Params.Get("gamma", scale ? OpGradReq.Write : OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(gamma_initializer), allow_deferred_init: true, differentiable: scale);
            this["beta"] = Params.Get("beta", center ? OpGradReq.Write : OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(beta_initializer), allow_deferred_init: true, differentiable: center);
            this["running_mean"] = Params.Get("running_mean", OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(running_mean_initializer), allow_deferred_init: true, differentiable: false);
            this["running_var"] = Params.Get("running_var", OpGradReq.Null, new Shape(in_channels),
                init: Initializer.Get(running_variance_initializer), allow_deferred_init: true, differentiable: false);
        }

        public int Axis { get; set; }
        public float Momentum { get; }
        public float Epsilon { get; }
        public bool Center { get; }
        public bool Scale { get; }
        public bool FuseRelu { get; }
        public bool FixGamma => !Scale;
        public bool Use_Global_Stats { get; set; }
        public int In_Channels { get; set; }
        public Parameter Gamma { get; set; }
        public Parameter Beta { get; set; }
        public Parameter RunningMean { get; set; }
        public Parameter RunningVar { get; set; }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            var (x, gamma, beta, running_mean, running_var) = args;

            if (FuseRelu)
            {
                if (x.IsNDArray)
                    return nd.Contrib.BatchNormWithReLU(x.NdX, gamma.NdX, beta.NdX, running_mean.NdX, running_var.NdX, eps: Epsilon, momentum: Momentum, axis: Axis, use_global_stats: Use_Global_Stats, fix_gamma: FixGamma);

                return sym.Contrib.BatchNormWithReLU(x.SymX, gamma.SymX, beta.SymX, running_mean.SymX, running_var.SymX, eps: Epsilon, momentum: Momentum, axis: Axis, use_global_stats: Use_Global_Stats, fix_gamma: FixGamma, symbol_name: "fwd");
            }
            else
            {
                if (x.IsNDArray)
                    return nd.BatchNorm(x.NdX, gamma.NdX, beta.NdX, running_mean.NdX, running_var.NdX, eps: Epsilon, momentum: Momentum, axis: Axis, use_global_stats: Use_Global_Stats, fix_gamma: FixGamma);

                return sym.BatchNorm(x.SymX, gamma.SymX, beta.SymX, running_mean.SymX, running_var.SymX, eps: Epsilon, momentum: Momentum, axis: Axis, use_global_stats: Use_Global_Stats, fix_gamma: FixGamma, symbol_name: "fwd");
            }
        }

        public override string ToString()
        {
            var in_channels = Params["gamma"].Shape[0];
            return $"{GetType().Name}(axis={Axis}, eps={Epsilon}, momentum={Momentum}, fix_gamma={!Scale}, use_global_stats={Use_Global_Stats}, in_channels={(in_channels > 0 ? in_channels.ToString() : "None")})";
        }
    }
}