using System;
using System.Collections.Generic;
using System.Text;
using MxNet;

namespace MxNet.NN.Initializers
{
    public class Constant : BaseInitializer
    {
        public float Value { get; set; }

        public Constant(float value) : base("constant")
        {
            Value = value;
        }

        public override void Generate(NDArray x)
        {
            x.Constant(this.Value);
        }

    }
}
