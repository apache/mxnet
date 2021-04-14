using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Initializers
{
    public class RandomNormal : BaseInitializer
    {
        public float Mean { get; set; }

        public float StdDev { get; set; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f) : base("random_normal")
        {
            Mean = mean;
            StdDev = stddev;
        }

        public override void Generate(NDArray x)
        {
            x.SampleGaussian(Mean, StdDev);
        }

    }
}
