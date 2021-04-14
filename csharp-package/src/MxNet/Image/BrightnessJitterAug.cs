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
using NumpyDotNet;

namespace MxNet.Image
{
    public class BrightnessJitterAug : Augmenter
    {
        public BrightnessJitterAug(float brightness)
        {
            Brightness = brightness;
        }

        public float Brightness { get; set; }

        public override NDArray Call(NDArray src)
        {
            var alpha = 1f + nd.Random.Uniform((int)-Brightness, (int)Brightness);
            src *= alpha;
            return src;
        }
    }
}