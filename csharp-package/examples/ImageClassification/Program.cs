using System;
using System.Linq;
using MxNet;
using MxNet.Gluon.ModelZoo.Vision;
using MxNet.Image;

namespace ImageClassification
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            AlexnetGluon.Run();
            //ResnetModule.Run();
        }
    }
}