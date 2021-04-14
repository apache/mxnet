using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Initializers
{
    public class VarianceScaling : BaseInitializer
    {
        public float Scale { get; set; }

        public string Mode { get; set; }

        public string Distribution { get; set; }

        public VarianceScaling(float scale = 1f, string mode = "fan_in", string distribution = "normal")
            : base("variance_scaling")
        {
            if (scale < 1f)
            {
                throw new ArgumentException("Scale must be positive value");
            }

            Util.ValidateParam("mode", mode, "fan_in", "fan_out", "fan_avg");
            Util.ValidateParam("distribution", distribution, "normal", "uniform");

            Scale = scale;
            Mode = mode;
            Distribution = distribution;
        }

        public override void Generate(NDArray x)
        {
            var hwScale = 1.0f;
            var shape = x.Shape;
            if (shape.Dimension > 2)
            {
                for (uint i = 2; i < shape.Dimension; ++i)
                    hwScale *= shape[i];
            }

            var @in = shape[1] * hwScale;
            var @out = shape[0] * hwScale;
            var factor = 1.0f;
            switch (Mode)
            {
                case "fan_avg":
                    factor = (@in + @out) / 2.0f;
                    break;
                case "fan_in":
                    factor = @in;
                    break;
                case "fan_out":
                    factor = @out;
                    break;
            }

            
            switch (Distribution)
            {
                case "uniform":
                    float limit = (float)Math.Sqrt(3f * Scale);
                    x.SampleUniform(-limit, limit);
                    break;
                case "normal":
                    float stddev = (float)Math.Sqrt(Scale) / 0.87962566103423978f;
                    x.SampleGaussian(0, stddev);
                    break;
            }
        }

    }
}
