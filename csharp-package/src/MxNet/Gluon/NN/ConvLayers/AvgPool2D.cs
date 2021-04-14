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

namespace MxNet.Gluon.NN
{
    public class AvgPool2D : _Pooling
    {
        public AvgPool2D((int, int)? pool_size = null, (int, int)? strides = null, (int, int)? padding = null,
            string layout = "NCHW",
            bool ceil_mode = false)
            : base(!pool_size.HasValue ? new[] {2, 2} : new[] {pool_size.Value.Item1, pool_size.Value.Item2}
                , strides.HasValue
                    ? new[] {strides.Value.Item1, strides.Value.Item2}
                    : new[] {pool_size.Value.Item1, pool_size.Value.Item2}
                , padding.HasValue ? new[] {padding.Value.Item1, padding.Value.Item2} : new[] { 0, 0 }
                , ceil_mode, false, PoolingType.Avg, layout, null)
        {
            if (layout != "NCHW" && layout != "NHWC")
                throw new Exception("Only NCHW and NHWC layouts are valid for 2D Pooling");
        }
    }
}