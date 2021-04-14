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
    public class AvgPool1D : _Pooling
    {
        public AvgPool1D(int pool_size = 2, int? strides = null, int padding = 0, string layout = "NCW",
            bool ceil_mode = false)
            : base(new[] {pool_size}
                , strides.HasValue ? new[] {strides.Value} : new[] {pool_size}
                , new[] {padding}, ceil_mode, false, PoolingType.Avg, layout, null)
        {
            if (layout != "NCW" && layout != "NWC")
                throw new Exception("Only NCW and NWC layouts are valid for 1D Pooling");
        }
    }
}