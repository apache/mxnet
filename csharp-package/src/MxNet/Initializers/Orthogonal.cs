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
using MxNet.Numpy;

namespace MxNet.Initializers
{
    public class Orthogonal : Initializer
    {
        public Orthogonal(float scale = 1.414f, string rand_type = "uniform")
        {
            Scale = scale;
            RandType = rand_type;
        }

        public float Scale { get; set; }

        public string RandType { get; set; }

        public override void InitWeight(string name, ref ndarray arr)
        {
            var nout = arr.shape[0];
            var nin = 1;
            ndarray tmp = null;
            ndarray res = null;
            for (var i = 1; i < arr.ndim; i++)
                nin *= arr.shape[i];

            if (RandType == "uniform")
                tmp = nd.Random.Uniform(-1, 1, new Shape(nout, nin));
            else if (RandType == "notmal")
                tmp = nd.Random.Normal(0, 1, new Shape(nout, nin));

            var (u, v) = nd.LinalgSyevd(tmp); //ToDo: use np.linalg.svd
            if (u.Shape == v.Shape)
                res = u;
            else
                res = v;

            res = Scale * res.reshape(arr.shape);
            arr = res;
        }
    }
}