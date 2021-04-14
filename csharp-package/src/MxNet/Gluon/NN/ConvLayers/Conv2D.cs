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
    public class Conv2D : _Conv
    {
        public Conv2D(int channels, (int, int) kernel_size, (int, int)? strides = null, (int, int)? padding = null,
            (int, int)? dilation = null, int groups = 1, string layout = "NCHW", int in_channels = 0,
            ActivationType? activation = null, bool use_bias = true, Initializer weight_initializer = null,
            string bias_initializer = "zeros")
            : base(channels, new[] {kernel_size.Item1, kernel_size.Item2},
                !strides.HasValue ? new[] {1, 1} : new[] {strides.Value.Item1, strides.Value.Item2},
                !padding.HasValue ? new[] {0, 0} : new[] {padding.Value.Item1, padding.Value.Item2},
                !dilation.HasValue ? new[] {1, 1} : new[] {dilation.Value.Item1, dilation.Value.Item2},
                groups, layout, in_channels, activation, use_bias,
                weight_initializer, bias_initializer, null, "Convolution")
        {
        }
    }
}