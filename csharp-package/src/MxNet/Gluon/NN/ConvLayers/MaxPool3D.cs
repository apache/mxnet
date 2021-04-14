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
    public class MaxPool3D : _Pooling
    {
        public MaxPool3D((int, int, int)? pool_size = null, (int, int, int)? strides = null,
            (int, int, int)? padding = null, string layout = "NCDHW",
            bool ceil_mode = false)
            : base(!pool_size.HasValue
                    ? new[] {2, 2, 2}
                    : new[] {pool_size.Value.Item1, pool_size.Value.Item2, pool_size.Value.Item3}
                , strides.HasValue
                    ? new[] {strides.Value.Item1, strides.Value.Item2, strides.Value.Item3}
                    : new[] {pool_size.Value.Item1, pool_size.Value.Item2, pool_size.Value.Item3}
                , !padding.HasValue
                    ? new[] {0, 0, 0}
                    : new[] {padding.Value.Item1, padding.Value.Item2, padding.Value.Item3}
                , ceil_mode, false, PoolingType.Max, layout, null)
        {
            if (layout != "NCDHW" && layout != "NDHWC")
                throw new Exception("Only NCDHW and NDHWC layouts are valid for 3D Pooling");
        }
    }
}