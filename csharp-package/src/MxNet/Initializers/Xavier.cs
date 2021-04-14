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
using System;

namespace MxNet.Initializers
{
    public class Xavier : Initializer
    {
        public Xavier(string rnd_type = "uniform", string factor_type = "avg", float magnitude = 3)
        {
            RndType = rnd_type;
            FactorType = factor_type;
            Magnitude = magnitude;
        }

        public string RndType { get; set; }

        public string FactorType { get; set; }

        public float Magnitude { get; set; }

        public override void InitWeight(string name, ref ndarray arr)
        {
            var shape = arr.shape;
            float hw_scale = 1;
            if (shape.Dimension < 2)
                throw new ArgumentException(
                    string.Format("Xavier initializer cannot be applied to vector {0}. It requires at least 2D", name));

            var fan_in = shape[1] * hw_scale;
            var fan_out = shape[0] * hw_scale;
            float factor = 1;
            if (FactorType == "avg")
                factor = (fan_in + fan_out) / 2;
            else if (FactorType == "in")
                factor = fan_in;
            else if (FactorType == "out")
                factor = fan_out;
            else
                throw new ArgumentException("Incorrect factor type");

            var scale = (float) Math.Sqrt(Magnitude / factor);

            if (RndType == "uniform")
                arr = nd.Random.Uniform(-scale, scale, arr.shape);
            else if (RndType == "gaussian")
                arr = nd.Random.Normal(0, scale, arr.shape);
            else
                throw new ArgumentException("Unknown random type");
        }

        public Symbol InitWeight(string name, Shape shape = null)
        {
            float hw_scale = 1;
            if (shape.Dimension < 2)
                throw new ArgumentException(
                    string.Format("Xavier initializer cannot be applied to vector {0}. It requires at least 2D", name));

            var fan_in = shape[1] * hw_scale;
            var fan_out = shape[0] * hw_scale;
            float factor = 1;
            if (FactorType == "avg")
                factor = (fan_in + fan_out) / 2;
            else if (FactorType == "in")
                factor = fan_in;
            else if (FactorType == "out")
                factor = fan_out;
            else
                throw new ArgumentException("Incorrect factor type");

            var scale = (float)Math.Sqrt(Magnitude / factor);

            if (RndType == "uniform")
                return sym.RandomUniform(-scale, scale, shape);
            else if (RndType == "gaussian")
                return sym.RandomNormal(0, scale, shape);
            else
                throw new ArgumentException("Unknown random type");
        }
    }
}