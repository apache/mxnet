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

namespace MxNet.Image
{
    public class DetRandomPadAug : DetAugmenter
    {
        public DetRandomPadAug((float, float)? aspect_ratio_range = null, (float, float)? area_range = null,
            int max_attempts = 50)
        {
            throw new NotImplementedException();
        }

        public override (NDArray, NDArray) Call(NDArray src, NDArray label)
        {
            throw new NotImplementedException();
        }

        private NDArray UpdateLabels(NDArray label, float[] pad_box, int height, int width)
        {
            throw new NotImplementedException();
        }

        private (float, float, float, float, NDArray) RandomCropProposal(NDArray label, int height, int width)
        {
            throw new NotImplementedException();
        }
    }
}