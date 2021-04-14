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
    public class RandomGrayAug : Augmenter
    {
        private readonly NDArray mat;

        public RandomGrayAug(float p)
        {
            Probability = p;
            mat = new NDArray(new[] {0.21f, 0.21f, 0.21f, 0.72f, 0.72f, 0.72f, 0.07f, 0.07f, 0.07f}).Reshape(3, 3);
        }

        public float Probability { get; set; }

        public override NDArray Call(NDArray src)
        {
            if (new Random().NextDouble() < Probability)
                src = nd.Dot(src, mat);

            return src;
        }
    }
}