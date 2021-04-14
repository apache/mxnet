using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Initializers
{
    public class RandomUniform : BaseInitializer
    {
        public float MinVal { get; set; }

        public float MaxVal { get; set; }

        public RandomUniform(float minval = 0f, float maxval = 0.05f) : base("random_uniform")
        {
            MinVal = minval;
            MaxVal = maxval;
        }

        public override void Generate(NDArray x)
        {
            x.SampleUniform(MinVal, MaxVal);
        }

    }
}
