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
using NumpyDotNet;

namespace MxNet.Image
{
    public class HueJitterAug : Augmenter
    {
        public HueJitterAug(float hue)
        {
            Hue = hue;
            Tyiq = new NDArray(new[] {0.299f, 0.587f, 0.114f, 0.596f, -0.274f, -0.321f, 0.211f, -0.523f, 0.311f})
                .Reshape(3, 3);
            ITyiq = new NDArray(new[] {1, 0.956f, 0.621f, 1, -0.272f, -0.647f, 1, -1.107f, 1.705f}).Reshape(3, 3);
        }

        public float Hue { get; }

        public NDArray Tyiq { get; set; }

        public NDArray ITyiq { get; set; }

        public override NDArray Call(NDArray src)
        {
            float alpha = FloatRnd.Uniform(-Hue, Hue);
            var u = (float) Math.Cos(alpha * Math.PI);
            var w = (float) Math.Sin(alpha * Math.PI);
            var bt = new NDArray(new[] {1, 0, 0, 0, u, -w, 0, w, u}).Reshape(3, 3);
            var t = nd.Dot(nd.Dot(ITyiq, bt), Tyiq).Transpose();
            src = nd.Dot(src, t);
            return src;
        }
    }
}