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
    public class Conv1DTranspose : _Conv
    {
        public Conv1DTranspose(int channels, int kernel_size, int strides = 1, int padding = 0, int output_padding = 0,
            int dilation = 1, int groups = 1, string layout = "NCW",
            ActivationType? activation = null, bool use_bias = true, Initializer weight_initializer = null,
            string bias_initializer = "zeros", int in_channels = 0)
            : base(channels, new[] {kernel_size}, new[] {strides}, new[] {padding},
                new[] {dilation}, groups, layout, in_channels, activation, use_bias,
                weight_initializer, bias_initializer, null, "Convolution")
        {
        }
    }
}