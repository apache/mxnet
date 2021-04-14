using MxNet;
using MxNet.Keras.Engine;
using MxNet.Keras.Layers;
using MxNet.Keras.Optimizers;
using System;

namespace MxNetKerasExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            NDArray x = new NDArray(new float[] { 0, 0, 0, 1, 1, 0, 1, 1 }).Reshape(4, 2);
            NDArray y = new NDArray(new float[] { 0, 1, 1, 0 }).Reshape(4, 1);

            var model = new Sequential();
            model.Add(new Dense(4, "relu", input_dim: 2));
            model.Add(new Dense(4, "relu"));
            model.Add(new Dense(1));

            model.Compile(new SGD(), "binary_crossentropy", new string[] { "binary_accuracy" });
            model.Fit(x, y, 2, 100);
        }
    }
}
