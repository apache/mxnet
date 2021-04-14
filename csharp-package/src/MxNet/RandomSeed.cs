using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet
{
    public static class FloatRnd
    {
        public static Random rnd = new Random();

        public static float Uniform(float low, float high)
        {
            return (float)rnd.NextDouble() * (high - low) + low;
        }
    }

    public class IntRnd
    {
        public static Random rnd = new Random();

        public static int Uniform(int low, int high)
        {
            return rnd.Next(low, high);
        }
    }
}
