using System;
using MxNet;
using MxNet.Gluon;
using MxNet.Gluon.NN;
using MxNet.Image;
using NumpyDotNet;
using OpenCvSharp;

namespace ConsoleTest
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            NDArray x = new NDArray(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Console.ReadLine();
        }
    }
}