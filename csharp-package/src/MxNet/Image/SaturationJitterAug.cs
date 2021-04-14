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
    public class SaturationJitterAug : Augmenter
    {
        private readonly NDArray coef;

        public SaturationJitterAug(float saturation)
        {
            Saturation = saturation;
            coef = new NDArray(new[] {0.299f, 0.587f, 0.114f}).Reshape(1, 3);
        }

        public float Saturation { get; set; }

        public override NDArray Call(NDArray src)
        {
            var alpha = 1f + FloatRnd.Uniform(-Saturation, Saturation);
            var gray = src * coef;
            gray = nd.Sum(gray, 2, true);
            gray *= 1 - alpha;
            src *= gray;
            src += gray;
            return src;
        }
    }
}