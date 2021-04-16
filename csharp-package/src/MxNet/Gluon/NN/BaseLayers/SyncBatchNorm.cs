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
using System;
using MxNet.Gluon.NN;

namespace MxNet.Gluon.NN
{
    public class SyncBatchNorm : _BatchNorm
    {
        public int? num_devices;

        public SyncBatchNorm(int in_channels = 0, int? num_devices = null, float momentum = 0.9f, float epsilon = 1e-5f,
            bool center = true, bool scale = true, bool use_global_stats = false, string beta_initializer = "zeros",
            string gamma_initializer = "ones", string running_mean_initializer = "zeros",
            string running_variance_initializer = "ones")
            : base(1, momentum, epsilon, center, scale, false, use_global_stats, beta_initializer, gamma_initializer,
                running_mean_initializer, running_variance_initializer, in_channels)
        {
            this.num_devices = num_devices;
        }

        internal int GetNumDevices()
        {
            Logger.Warning("Caution using SyncBatchNorm: if not using all the GPUs, please mannually set num_devices");
            var num_devices = MxUtil.GetGPUCount();
            num_devices = num_devices > 0 ? num_devices : 1;
            return num_devices;
        }

        public override NDArrayOrSymbolList HybridForward(NDArrayOrSymbolList args)
        {
            var (x, gamma, beta, running_mean, running_var) = args;

            if (x.IsNDArray)
                return nd.Contrib.SyncBatchNorm(x, gamma, beta, running_mean, running_var, "", Epsilon, Momentum, FixGamma,
                     Use_Global_Stats, false, num_devices.HasValue ? num_devices.Value : 1);

            return sym.Contrib.SyncBatchNorm(x, gamma, beta, running_mean, running_var, "", Epsilon, Momentum, FixGamma,
                     Use_Global_Stats, false, num_devices.HasValue ? num_devices.Value : 1, "fwd");
        }
    }
}