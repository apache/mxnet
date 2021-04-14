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
namespace MxNet.Image
{
    public class LightingAug : Augmenter
    {
        public LightingAug(float alphastd, NDArray eigval, NDArray eigvec)
        {
            Alphastd = alphastd;
            Eigval = eigval;
            Eigvec = eigvec;
        }

        public float Alphastd { get; set; }

        public NDArray Eigval { get; set; }

        public NDArray Eigvec { get; set; }

        public override NDArray Call(NDArray src)
        {
            var alpha = nd.Random.Uniform(0, Alphastd, new Shape(3));
            NDArray rgb = nd.Dot(Eigvec * alpha, Eigval);
            src += rgb;
            return src;
        }
    }
}