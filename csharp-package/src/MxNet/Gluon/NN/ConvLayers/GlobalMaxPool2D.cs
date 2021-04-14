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
    public class GlobalMaxPool2D : _Pooling
    {
        public GlobalMaxPool2D(string layout = "NCHW")
            : base(new[] {1, 1}, null, new[] {0, 0}, true, true, PoolingType.Max, layout, null)
        {
            if (layout != "NCHW" && layout != "NHWC")
                throw new Exception("Only NCHW and NHWC layouts are valid for 2D Pooling");
        }
    }
}